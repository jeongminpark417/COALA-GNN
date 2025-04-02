#pragma once


#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#include "util.h"
#include "host_util.h"
#include "nvm_types.h"
#include "nvm_util.h"
#include "buffer.h"
#include "ctrl.h"
#include <iostream>
#include "nvm_parallel_queue.h"
#include "nvm_cmd.h"

#include "seqlock.h"
#include "nvshmem.h"
#include "nvshmemx.h"

#define FULL_MASK 0xFFFFFFFF
#define FF_64 0xFFFFFFFFFFFFFFFF
#define FE_64 0xFFFEFFFFFFFFFFFF

#define FF_48 0xFFFFFFFFFFFFFF
#define FF_16 0xFFFF
#define FE_16 0xFFFE


template <typename T = float>
 __forceinline__
__device__
void warp_memcpy(void* src, void* dst, size_t size, uint32_t mask){
     T* src_ptr = (T*) src;
     T* dst_ptr = (T*) dst;
     
     uint32_t count = __popc(mask);
     uint32_t lane_id = threadIdx.x %32;
     
     uint32_t my_id = count - (__popc(mask>>(lane_id)));
     for(; my_id < size/sizeof(T); my_id += count){
          dst_ptr[my_id] =  src_ptr[my_id]; 
     }
 }


template<typename T>
struct NVSHMEM_cache_d_t {

    unsigned num_gpus, my_GPU_id_;
    seqlock* set_locks_, *way_locks_;
    uint64_t num_sets, num_ways, CL_SIZE;

    uint64_t* keys_;
    uint32_t* set_cnt_;

    Controller** d_ctrls_;
    uint32_t n_ctrls_;
    uint64_t n_blocks_per_page_;

    uint8_t* base_addr_;
    bool prps_;
    uint64_t* prp1_, *prp2_;                  


    simt::atomic<uint64_t, simt::thread_scope_device>* q_head, *q_tail, *q_lock, *extra_reads;
    simt::atomic<uint64_t, simt::thread_scope_device> double_read, hit_cnt, miss_cnt;


    uint8_t evict_policy = 0;

    bool color_track_;
    int32_t* color_counters;
    uint64_t* color_meta_;
    int64_t* node_color_buffer_;

    bool SSD_SIM = true;
    void* sim_buf = nullptr;

    NVSHMEM_cache_d_t(seqlock* set_locks, seqlock* way_locks,uint64_t n_sets, uint64_t n_ways, uint64_t cl_size, uint64_t* keys, uint32_t* set_cnt,
                 Controller** d_ctrls, uint32_t n_ctrls, uint64_t n_blocks_per_page, uint8_t* base_addr, uint64_t* prp1, uint64_t* prp2, bool prps,
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_head, simt::atomic<uint64_t, simt::thread_scope_device>* queue_tail, 
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_lock, simt::atomic<uint64_t, simt::thread_scope_device>* queue_extra_reads,
                // bool color_track,
                // int32_t* color_counter, uint64_t* color_meta, int64_t* node_color_buffer,
                 unsigned int my_GPU_id, unsigned n_gpus, 
                 bool is_simulation, float* sim_b
                 ) :
        set_locks_(set_locks),
        way_locks_(way_locks),
        num_sets(n_sets),
        num_ways(n_ways),
        CL_SIZE(cl_size),
        keys_(keys),
        set_cnt_(set_cnt),
        base_addr_(base_addr),
        prp1_(prp1),
        prp2_(prp2),
        prps_(prps),
        d_ctrls_(d_ctrls),
        n_ctrls_(n_ctrls),
        n_blocks_per_page_(n_blocks_per_page),
        q_head(queue_head),
        q_tail(queue_tail),
        q_lock(queue_lock),
        extra_reads(queue_extra_reads),
        
        // color_track_(color_track),
        // color_counters(color_counter),
        // color_meta_(color_meta),
        // node_color_buffer_(node_color_buffer),

        my_GPU_id_(my_GPU_id),
        num_gpus(n_gpus),
        SSD_SIM(is_simulation),
        sim_buf(sim_b)
 
        {
            double_read = 0;
            hit_cnt = 0;
            miss_cnt = 0;
        }

    // __forceinline__
    // __device__
    // void
    // print_stats(){
    //     printf("GPU hit count: %llu CPU hit count: %llu\n", (unsigned long long) gpu_hit_cnt,  (unsigned long long) cpu_hit_cnt);            
    //     printf("GPU miss count: %llu CPU miss count: %llu\n", (unsigned long long) gpu_miss_cnt,  (unsigned long long) cpu_miss_cnt);            

    //     uint64_t system_hit = gpu_hit_cnt + cpu_hit_cnt;
    //     uint64_t system_miss = gpu_miss_cnt + cpu_miss_cnt;
        
    //     float overall_hit_ratio = (float) (system_hit) / (system_hit + system_miss);
    //     float gpu_hit_ratio = (float) (gpu_hit_cnt) / (system_hit + system_miss);
    //     float cpu_hit_ratio = (float) (cpu_hit_cnt) / (system_hit + system_miss);

    //     printf("System Hit ratio: %f GPU hit ratio: %f CPU hit ratio:%f\n", overall_hit_ratio, gpu_hit_ratio, cpu_hit_ratio);

    //     gpu_hit_cnt = 0;
    //     gpu_miss_cnt = 0;
    //     cpu_hit_cnt = 0;
    //     cpu_miss_cnt = 0;
    // }



    __forceinline__
    __device__
    unsigned 
    search_ways(uint64_t tag_id, uint32_t thread_mask, uint64_t set_id){

        bool found = false;
        unsigned way = num_ways;
        uint64_t way_offset = set_id * num_ways;
        uint64_t* cur_ways = keys_ + way_offset;

        unsigned effective_warp_size = __popc(thread_mask);
        uint32_t lane = lane_id();
        uint32_t effective_lane_id = __popc(thread_mask & ((1 << lane) - 1));
        uint32_t effective_warp_leader = __ffs(thread_mask) - 1;
        
        __syncwarp(thread_mask);

        for(uint32_t i = effective_lane_id; i < num_ways; i += effective_warp_size){
            found = tag_id == cur_ways[i];
            if(found){
                way = i;
                break;
            }
        }
        unsigned found_mask = __ballot_sync(thread_mask, found);
        unsigned leader = found_mask ? (__ffs(found_mask) - 1) : (__ffs(thread_mask) - 1);

        way = __shfl_sync(thread_mask, way, leader);
        return way;
    }

    __forceinline__
    __device__
    uint64_t get_cl_id(uint64_t key){
        uint64_t cl_id = key / CL_SIZE;
        return cl_id;
    }
    
    __forceinline__
    __device__
    uint64_t get_set_id(uint64_t key){
        uint64_t set_id = key % num_sets;
        return set_id;
    }

    __forceinline__
    __device__
    uint64_t get_dist_set_id(uint64_t key, int num_gpus){
        uint64_t set_id = (key / num_gpus) % num_sets;
        return set_id;
    }

    __forceinline__
    __device__
    uint32_t 
    round_robin_evict(uint32_t lane, uint32_t leader, uint32_t mask, uint64_t set_id){
        uint32_t way = 0;
        if(lane == leader){
            way = (set_cnt_[set_id]++) % num_ways;
            
        }
        else{
            way =  0;
        }
        way = __shfl_sync(mask, way, leader);
        return way;
    }

 

    __forceinline__
    __device__
    uint32_t evict(uint32_t lane, uint32_t leader, uint32_t mask, uint64_t set_id, uint8_t eviction_policy=0){
        uint32_t evict_way;
        switch(eviction_policy){
            //RR
            case 0 :
                evict_way = round_robin_evict(lane, leader, mask, set_id);
                return evict_way;


            default:
                return 0;
        }

    }

    inline 
    __device__ 
    void enqueue_second( QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head) {
        nvm_cmd_rw_blks(cmd, starting_lba, 1);
        unsigned int ns = 8;
        do {
            uint64_t cur_pc_head = q_head->load(simt::memory_order_relaxed);
            bool sec = ((cur_pc_head < pc_prev_head) && (pc_prev_head <= pc_pos)) ||
                ((pc_prev_head <= pc_pos) && (pc_pos < cur_pc_head)) ||
                ((pc_pos < cur_pc_head) && (cur_pc_head < pc_prev_head));

            if (sec) break;

            //if not
            uint64_t qlv = q_lock->load(simt::memory_order_relaxed);
            //got lock
            if (qlv == 0) {
                qlv = q_lock->fetch_or(1, simt::memory_order_acquire);
                if (qlv == 0) {
                    uint64_t cur_pc_tail;// = pc->q_tail.load(simt::memory_order_acquire);

                    uint16_t sq_pos = sq_enqueue(&qp->sq, cmd, q_tail, &cur_pc_tail);
                    uint32_t head, head_;
                    uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

                    q_head->store(cur_pc_tail, simt::memory_order_release);
                    q_lock->store(0, simt::memory_order_release);
                    extra_reads->fetch_add(1, simt::memory_order_relaxed);
                    cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);



                    break;
                }
            }
            #if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
                    __nanosleep(ns);
                    if (ns < 256) {
                        ns *= 2;
                    }
            #endif    
            } while(true);

    }
   
    inline 
    __device__ 
    void read_data(QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {

        nvm_cmd_t cmd;
        uint16_t cid = get_cid(&(qp->sq));

        nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
        uint64_t prp1, prp2;
      
        prp1 = prp1_[pc_entry];
        prp2 = 0;
        if (prps_)
            prp2 = prp2_[pc_entry];
    
        nvm_cmd_data_ptr(&cmd, prp1, prp2);
        nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
        uint16_t sq_pos = sq_enqueue(&qp->sq, &cmd);
        uint32_t head, head_;
        uint64_t pc_pos;
        uint64_t pc_prev_head;

        uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

        qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
        pc_prev_head = q_head->load(simt::memory_order_relaxed);
        pc_pos = q_tail->fetch_add(1, simt::memory_order_acq_rel);

        cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);
        enqueue_second(qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head);

        put_cid(&qp->sq, cid);
    }

    __forceinline__
    __device__
    void read_page(uint64_t pg_id, uint64_t evicted_page, uint32_t queue){

        uint64_t ctrl = pg_id % n_ctrls_;
        uint64_t b_page = pg_id / n_ctrls_;

        Controller* c = d_ctrls_[ctrl];
        c->access_counter.fetch_add(1, simt::memory_order_relaxed);
        read_data((c->d_qps)+queue, ((b_page)*n_blocks_per_page_), n_blocks_per_page_, evicted_page);
        return;        
    }
     __forceinline__
    __device__
    void read_page_simulation(uint64_t pg_id, void* dst, uint32_t mask){
        __nanosleep(1000*10);
        void* src = sim_buf + (pg_id) * CL_SIZE;
        warp_memcpy<float>(src, dst, CL_SIZE, mask);
        return;
    }
                


    __forceinline__
    __device__
    void 
    get_data(uint64_t id, T* output_ptr, int rank, int dst_gpu){

        uint32_t lane = lane_id();
        uint32_t mask = __activemask();

        //uint64_t cl_id = id / num_gpus;
        uint64_t cl_id = id ;

        uint64_t set_id = get_dist_set_id(cl_id, num_gpus);
        uint64_t set_offset = set_id * (num_ways);
        seqlock* cur_set_lock = set_locks_ + set_id;
        seqlock* cur_cl_seqlock = way_locks_ + set_id * (num_ways );

        uint32_t warp_leader = __ffs(mask) - 1;
        bool cont = true;
        bool done = false;

        do {
            uint64_t set_before, set_after;
            if(lane == warp_leader) {
                set_before = cur_set_lock->read_lock();
            }
            set_before = __shfl_sync(mask, set_before, warp_leader);
            unsigned way = search_ways(cl_id, mask, set_id);

            if(way < num_ways){
                if(way < num_ways){

                    seqlock* way_lock = way_locks_ + set_offset + way;
                    uint64_t way_before;
                    if(lane == warp_leader) {
                        way_before = way_lock->read_busy_block();
                    }
                    way_before = __shfl_sync(mask, way_before, warp_leader);

                    bool hit = false;
                    auto way_offset = set_offset + way;
                    auto way_key = keys_ + way_offset;

                    if(cl_id == (*way_key)){
                        hit = true;
                        void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
                        nvshmemx_float_put_warp(output_ptr, (float*) src, CL_SIZE/sizeof(float), dst_gpu);
                    }

                    __syncwarp(mask);

                    uint64_t way_after;
                    if(lane == warp_leader) {
                        way_after = way_lock->read_busy_unlock();
                    }
                    way_after = __shfl_sync(mask, way_after, warp_leader);

                    if(!done)
                        done = hit && (way_before == way_after);

                    unsigned not_done_mask = __ballot_sync(mask, !done);
                    if(not_done_mask == 0){
                        if(lane == warp_leader)
                            hit_cnt.fetch_add(1, simt::memory_order_relaxed);
                        return;
                    }
                }
            }
            if(lane == warp_leader) set_after = cur_set_lock->read_unlock();
            set_after = __shfl_sync(mask, set_after, warp_leader);
            cont = set_before != set_after;
        } while(cont);

        //miss
        
        if(lane == warp_leader) {
            cur_set_lock->write_lock();
        }
        __syncwarp(mask);

        //EVICTION
        uint32_t way;
        way = evict(lane, warp_leader, mask, set_id, evict_policy);

        if(lane == warp_leader) {
            // if(color_track_){
            //     int32_t prev = atomicSub(&(color_counters[color_meta_[set_offset + way]]), 1); 
            // }
            (cur_cl_seqlock + way) -> write_busy_lock();
        }
        //Check
        uint64_t old_key =  keys_[set_offset + way];
        keys_[set_offset + way] = cl_id;

        // if(color_track_){
        //     int64_t color = node_color_buffer_[cl_id]; 
        //     color_meta_[set_offset + way] = color;
        //     if(lane == warp_leader) {
        //         atomicAdd(&(color_counters[color]), 1); 
        //     }
        // }

        if(lane == warp_leader) {
            cur_set_lock->write_unlock();
        }
        __syncwarp(mask);
        
        if (SSD_SIM){
            void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
            read_page_simulation(cl_id, src, mask);
        }
        else{
            if(lane == warp_leader) {
                uint32_t queue = get_smid() % (d_ctrls_[0]->n_qps);
                read_page(cl_id, set_offset + way, queue);
            }
        }

        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_unbusy();
        }
        __syncwarp(mask);

        void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
        nvshmemx_float_put_warp(output_ptr, (float*) src, CL_SIZE/sizeof(float), dst_gpu);
    
        if(lane == warp_leader) 
            miss_cnt.fetch_add(1, simt::memory_order_relaxed);
        
        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_busy_unlock();
        }
        __syncwarp(mask);
    }

};



template<typename T>
struct NVSHMEM_cache_handle{
    NVSHMEM_cache_d_t<T>* cache_ptr = nullptr;

    seqlock* d_set_locks, *d_way_locks;
    uint64_t num_sets_, num_ways_, CL_SIZE_;

    uint64_t* keys_;
    uint32_t* set_cnt_;

    Controller* ctrls_;

    DmaPtr pages_dma;
    bool prps;

    uint8_t* base_addr;
    BufferPtr prp1_buf, prp2_buf;

    BufferPtr page_ticket_buf, ctrl_counter_buf;
    BufferPtr q_head_buf, q_tail_buf, q_lock_buf, extra_reads_buf;

    BufferPtr double_reads_buf, d_ctrls_buff;

    unsigned my_GPU_id_;
    unsigned num_gpus = 1;
    bool track_color;

    bool SSD_SIM;
    float* sim_buf;
    // uint64_t num_color_= 1;
    // int32_t* color_counters = nullptr;
    // int32_t* device_color_counters = nullptr;
    // uint64_t* color_meta;

    __host__ 
    NVSHMEM_cache_handle(uint64_t num_sets, uint64_t  num_ways, uint64_t cl_size, const std::vector<Controller*>& ctrls, 
                         unsigned int my_GPU_id, unsigned n_gpus, bool track_color_flag, bool is_simulation, float* sim_b) :
    num_sets_(num_sets),
    num_ways_(num_ways),
    CL_SIZE_(cl_size),
    my_GPU_id_(my_GPU_id),
    num_gpus(n_gpus),
    track_color(track_color_flag),
    SSD_SIM(is_simulation),
    sim_buf(sim_b)
    {

        if(is_simulation){
            if(sim_b == nullptr){
                printf("Error: the data buffer for simulation is nullptr\n");
                return;
            }
        }
        auto cudaDevice = my_GPU_id;
        uint64_t* prp1, *prp2;

        cuda_err_chk(cudaMalloc((void**)&d_set_locks, sizeof(seqlock) * num_sets_));
        cuda_err_chk(cudaMalloc((void**)&d_way_locks, sizeof(seqlock) * num_sets_ * (num_ways_ )));

        cuda_err_chk(cudaMalloc((void**)&set_cnt_, sizeof(uint32_t) * num_sets_));
        cuda_err_chk(cudaMalloc((void**)&keys_, sizeof(uint64_t) * num_sets_ * (num_ways_ )));

        cuda_err_chk(cudaMemset(d_set_locks, 0, sizeof(seqlock) * num_sets_));
        cuda_err_chk(cudaMemset(d_way_locks, 0, sizeof(seqlock) * num_sets_ * (num_ways_ )));
        cuda_err_chk(cudaMemset(set_cnt_, 0, sizeof(uint32_t) * num_sets_));
        cuda_err_chk(cudaMemset(keys_, 0xFF, sizeof(uint64_t) * num_sets_ * (num_ways_ )));

        cuda_err_chk(cudaMalloc((void**)&cache_ptr, sizeof(NVSHMEM_cache_d_t<T>)));

       if(track_color){
            //printf("allocating color counters num_color %i \n", num_color_);
            // cuda_err_chk(cudaHostAlloc((int32_t **)&color_counters, sizeof(int32_t) * num_color_ , cudaHostAllocMapped));
            // cuda_err_chk(cudaHostGetDevicePointer((int32_t **)&device_color_counters, (int32_t *)color_counters, 0));
            // cuda_err_chk(cudaMemset(device_color_counters, 0x0, (uint64_t) sizeof(int32_t) * num_color_));

            // cuda_err_chk(cudaMalloc((void**)&color_meta, sizeof(uint64_t) * num_sets_ * num_ways_));
            // cuda_err_chk(cudaMemset(color_meta, 0x0, sizeof(uint64_t) * num_sets_ * num_ways_));
        }
    
        uint64_t cache_size = CL_SIZE_ * num_sets_ * num_ways_;
        // Backend storage is CPU memory
        if (SSD_SIM){
            cuda_err_chk(cudaMalloc((void**)&base_addr, cache_size));
        }
        // Backend storage is NVMe SSDs
        else{
            const Controller& ctrl = ctrls[0][0]; 
            pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
            base_addr = (uint8_t*) pages_dma.get()->vaddr;
        }

        ctrl_counter_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_head_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_tail_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        q_lock_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        extra_reads_buf = createBuffer(sizeof(simt::atomic<uint64_t, simt::thread_scope_device>), cudaDevice);
        
        auto cache_ctrl_counter = (simt::atomic<uint64_t, simt::thread_scope_device>*)ctrl_counter_buf.get();

        auto cache_q_head = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_head_buf.get();
        auto cache_q_tail = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_tail_buf.get();
        auto cache_q_lock = (simt::atomic<uint64_t, simt::thread_scope_device>*)q_lock_buf.get();
        auto cache_extra_reads = (simt::atomic<uint64_t, simt::thread_scope_device>*)extra_reads_buf.get();


        uint32_t n_ctrls = ctrls.size();
        d_ctrls_buff = createBuffer(n_ctrls * sizeof(Controller*), cudaDevice);
        auto d_ctrls = (Controller**) d_ctrls_buff.get();
        uint64_t n_blocks_per_page = 0;
       

        if (SSD_SIM){
            prps = false;
        }
        else{
            prps = false;
            const Controller& ctrl = ctrls[0][0]; 
            n_blocks_per_page = (CL_SIZE_/ctrl.blk_size);
            for (size_t k = 0; k < n_ctrls; k++)
                cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
            uint64_t how_many_in_one = ctrl.ctrl->page_size/CL_SIZE_;

            if (CL_SIZE_ <= pages_dma.get()->page_size) {
                    this->prp1_buf = createBuffer(num_sets_ * num_ways_ * sizeof(uint64_t), cudaDevice);
                    prp1 = (uint64_t*) this->prp1_buf.get();
                    uint64_t* temp = new uint64_t[how_many_in_one * pages_dma.get()->n_ioaddrs];
                    std::memset(temp, 0, how_many_in_one *  pages_dma.get()->n_ioaddrs);

                    for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                        for (size_t j = 0; (j < how_many_in_one); j++) {
                            temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*CL_SIZE_;
                        }
                    }
                    cuda_err_chk(cudaMemcpy(prp1, temp, num_sets_ * num_ways_ * sizeof(uint64_t), cudaMemcpyHostToDevice));
                    delete temp;
                }
        }

        NVSHMEM_cache_d_t<T> cache_host(d_set_locks, d_way_locks, num_sets_, num_ways_, CL_SIZE_, keys_, set_cnt_, 
        d_ctrls, n_ctrls, n_blocks_per_page, base_addr, prp1, prp2, prps,
        cache_q_head, cache_q_tail, cache_q_lock, cache_extra_reads, 
        //color_track, device_color_counters, color_meta, node_color_buffer,      
        my_GPU_id_, num_gpus, 
        SSD_SIM, sim_buf);


        cuda_err_chk(cudaMemcpy(cache_ptr, &cache_host, sizeof(NVSHMEM_cache_d_t<T>), cudaMemcpyHostToDevice));
        std::cout << "Cache Device Setting Done\n";
    }

    __host__ 
    ~NVSHMEM_cache_handle() {
        cuda_err_chk(cudaFree(d_set_locks));
        cuda_err_chk(cudaFree(d_way_locks));
        cuda_err_chk(cudaFree(set_cnt_));
        cuda_err_chk(cudaFree(keys_));
        cuda_err_chk(cudaFree(cache_ptr));

        if (SSD_SIM){
            cuda_err_chk(cudaFree(base_addr));
        }
    }

    __host__ 
    NVSHMEM_cache_d_t<T>* 
    get_ptr(){
        return cache_ptr;
    }

    // int32_t* 
    // get_color_counter_ptr(){
    //     return color_counters;
    // }

};



