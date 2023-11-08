#ifndef __SET_ASSOCIATIVE_PAGE_CACHE_H__
#define __SET_ASSOCIATIVE_PAGE_CACHE_H__

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
//#include "window_buffer.h"

#define  RR 0x01

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



class seqlock {
    simt::atomic<uint32_t, simt::thread_scope_device>  ticket_;
    simt::atomic<uint32_t, simt::thread_scope_device>  current_;

    public:

    __device__
    unsigned 
    read_lock(){
        unsigned ticket = current_.load(simt::memory_order_acquire);
        return ticket & ~0x1;
    }

    __device__
    unsigned
    read_unlock(){
        return current_.fetch_or(0, simt::memory_order_release);
    }

    __device__
    unsigned 
    read_busy_block(){
        unsigned delay = 8;
        unsigned ticket = current_.load(simt::memory_order_acquire);
        while(ticket & 0x1){
            if(delay < 256) delay <<= 1;
            else delay = 8;

            ticket = current_.load(simt::memory_order_acquire);
        }
        return ticket & ~(0x1U << 1);
    }

    __device__
    unsigned 
    read_busy_unlock(){
        return current_.fetch_or(0, simt::memory_order_release) & ~(0X1U << 1);
    }

   

    __device__
    void
    busy_wait(unsigned ticket){
        unsigned current = current_.load(simt::memory_order_acquire);
        while(current != ticket){
            __nanosleep(8);
            current = current_.load(simt::memory_order_acquire);
        }
        current_.fetch_add(1, simt::memory_order_release);
    }

    __device__
    unsigned
    write_lock(){
        unsigned ticket = ticket_.fetch_add(2, simt::memory_order_acquire);
        busy_wait(ticket);
    }

    __device__
    unsigned
    write_unlock(){
        current_.fetch_add(1, simt::memory_order_release);
    }

    __device__
    unsigned
    write_busy_lock(){
        unsigned ticket = ticket_.fetch_add(4, simt::memory_order_acquire);
        busy_wait(ticket);
    }

    __device__
    unsigned
    write_unbusy(){
        current_.fetch_add(1, simt::memory_order_release);
    }

    __device__
    unsigned
    write_busy_unlock(){
        current_.fetch_add(2, simt::memory_order_release);
    }

};



template<typename T>
struct SA_cache_d_t {

    seqlock* set_locks_;
    seqlock* way_locks_;

    uint64_t num_sets;
    uint64_t num_ways;
    uint64_t CL_SIZE;


    uint64_t* keys_;
    uint32_t* set_cnt_;

    //need to be changed
    uint8_t evict_policy = RR;

    //nvme controllers
    Controller** d_ctrls_;
    uint32_t n_ctrls_;

    uint64_t n_blocks_per_page_;

    uint8_t* base_addr_;
    uint64_t* prp1_;                  
    uint64_t* prp2_;   
    bool prps_;


    simt::atomic<uint64_t, simt::thread_scope_device>* q_head;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_tail;
    simt::atomic<uint64_t, simt::thread_scope_device>* q_lock;
    simt::atomic<uint64_t, simt::thread_scope_device>* extra_reads;


    SA_cache_d_t(seqlock* set_locks, seqlock* way_locks,uint64_t n_sets, uint64_t n_ways, uint64_t cl_size, uint64_t* keys, uint32_t* set_cnt,
                 Controller** d_ctrls, uint32_t n_ctrls, uint64_t n_blocks_per_page, uint8_t* base_addr, uint64_t* prp1, uint64_t* prp2, bool prps,
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_head, simt::atomic<uint64_t, simt::thread_scope_device>* queue_tail, 
                 simt::atomic<uint64_t, simt::thread_scope_device>* queue_lock, simt::atomic<uint64_t, simt::thread_scope_device>* queue_extra_reads                 ) :
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
    extra_reads(queue_extra_reads)
    {}


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
    uint32_t 
    round_robin_evict(uint64_t set_id){
        return (set_cnt_[set_id]++) % num_ways;
    }

    __forceinline__
    __device__
    uint32_t evict(uint64_t set_id, uint8_t eviction_policy){
        switch(eviction_policy){
            case RR :
                uint32_t evict_way = round_robin_evict(set_id);
                return evict_way;

            default:
                return 0;
        }

    }



    inline 
    __device__ 
    void 
    enqueue_second( QueuePair* qp, const uint64_t starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos, const uint64_t pc_prev_head) {
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


    inline __device__ 
    void read_data(QueuePair* qp, const uint64_t starting_lba, const uint64_t n_blocks, const unsigned long long pc_entry) {

        nvm_cmd_t cmd;
        uint16_t cid = get_cid(&(qp->sq));

        nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
        uint64_t prp1 = prp1_[pc_entry];
        uint64_t prp2 = 0;
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

        //Need to be fixed (Assuming Page Start is 0)
        uint64_t ctrl = pg_id % n_ctrls_;
        uint64_t b_page = pg_id / n_ctrls_;



        Controller* c = d_ctrls_[ctrl];
        c->access_counter.fetch_add(1, simt::memory_order_relaxed);
		read_data((c->d_qps)+queue, ((b_page)*n_blocks_per_page_), n_blocks_per_page_, evicted_page);
        return;

    }

    __forceinline__
    __device__
    void 
    get_data(uint64_t cl_id, T* output_ptr){

        uint32_t lane = lane_id();
        uint32_t mask = __activemask();

        //uint64_t cl_id = get_cl_id(key);
        uint64_t set_id = get_set_id(cl_id);
        uint64_t set_offset = set_id * num_ways;

        seqlock* cur_set_lock = set_locks_ + set_id;
        seqlock* cur_cl_seqlock = way_locks_ + set_id * num_ways;

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
                        //memcpy
                        void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
                        warp_memcpy<T>(src, output_ptr, CL_SIZE, mask);
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
        uint32_t way = evict(set_id, evict_policy);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way) -> write_busy_lock();
        }
        __syncwarp(mask);

        //Check
        keys_[set_offset + way] = cl_id;

        if(lane == warp_leader) {
            cur_set_lock->write_unlock();
        }
        __syncwarp(mask);

        //HANDLE MISS
        //read_page();
        uint32_t queue;
        if(lane == warp_leader) {
            queue = get_smid() % (d_ctrls_[0]->n_qps);
        }
        queue = __shfl_sync(mask, queue, warp_leader);
        read_page(cl_id, set_offset + way, queue);

        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_unbusy();
        }
        __syncwarp(mask);

        void* src = ((void*)base_addr_)+ (set_offset + way) * CL_SIZE;
        warp_memcpy<T>(src, output_ptr, CL_SIZE, mask);
        __syncwarp(mask);

        if(lane == warp_leader) {
            (cur_cl_seqlock + way)->write_busy_unlock();
        }
        __syncwarp(mask);
        

    }
};



template<typename T>
struct GIDS_SA_handle{
    SA_cache_d_t<T>* cache_ptr = nullptr;

    seqlock* d_set_locks;
    seqlock* d_way_locks;

    uint64_t num_sets_;
    uint64_t num_ways_;
    uint64_t CL_SIZE_;


    uint64_t* keys_;
    uint32_t* set_cnt_;

    Controller* ctrls_;

    DmaPtr pages_dma;
    DmaPtr prp_list_dma;
    bool prps;

    uint8_t* base_addr;
    BufferPtr prp1_buf;
    BufferPtr prp2_buf; 


    BufferPtr page_ticket_buf;
    BufferPtr ctrl_counter_buf;
    BufferPtr q_head_buf;
    BufferPtr q_tail_buf;
    BufferPtr q_lock_buf;
    BufferPtr extra_reads_buf;

    BufferPtr d_ctrls_buff;



    __host__ 
    GIDS_SA_handle(uint64_t num_sets, uint64_t num_ways, uint64_t cl_size, const Controller& ctrl, const std::vector<Controller*>& ctrls,  const uint32_t cudaDevice) :
    num_sets_(num_sets),
    num_ways_(num_ways),
    CL_SIZE_(cl_size)
    {
        uint64_t* prp1;
        uint64_t* prp2;


        cudaMalloc((void**)&d_set_locks, sizeof(seqlock) * num_sets_);
        cudaMalloc((void**)&d_way_locks, sizeof(seqlock) * num_sets_ * num_ways_);

        cudaMalloc((void**)&set_cnt_, sizeof(uint32_t) * num_sets_);
        cudaMalloc((void**)&keys_, sizeof(uint64_t) * num_sets_ * num_ways_);

        cudaMemset(d_set_locks, 0, sizeof(seqlock) * num_sets_);
        cudaMemset(d_way_locks, 0, sizeof(seqlock) * num_sets_ * num_ways_);
        cudaMemset(set_cnt_, 0, sizeof(seqlock) * num_sets_);
        cudaMemset(keys_, 0xFF, sizeof(uint64_t) * num_sets_ * num_ways_);

        cudaMalloc((void**)&cache_ptr, sizeof(SA_cache_d_t<T>));

 
        uint64_t cache_size = CL_SIZE_ * num_sets_ * num_ways_;
        pages_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(cache_size, 1UL << 16), cudaDevice);
        base_addr = (uint8_t*) pages_dma.get()->vaddr;
        const uint32_t uints_per_page = ctrl.ctrl->page_size / sizeof(uint64_t);


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
        auto n_blocks_per_page = (CL_SIZE_/ctrl.blk_size);
      //  auto n_cachelines_for_states = np/STATES_PER_CACHELINE;
        
        for (size_t k = 0; k < n_ctrls; k++)
            cuda_err_chk(cudaMemcpy(d_ctrls+k, &(ctrls[k]->d_ctrl_ptr), sizeof(Controller*), cudaMemcpyHostToDevice));
        

        if (CL_SIZE_ <= pages_dma.get()->page_size) {
            std::cout << "Cond1\n";
            uint64_t how_many_in_one = ctrl.ctrl->page_size/CL_SIZE_;

            this->prp1_buf = createBuffer(num_sets_ * num_ways_ * sizeof(uint64_t), cudaDevice);
            prp1 = (uint64_t*) this->prp1_buf.get();

            std::cout << (num_sets_ * num_ways_) << " " << sizeof(uint64_t) << " " << how_many_in_one << " " << this->pages_dma.get()->n_ioaddrs <<std::endl;
            uint64_t* temp = new uint64_t[how_many_in_one * pages_dma.get()->n_ioaddrs];
            std::memset(temp, 0, how_many_in_one *  pages_dma.get()->n_ioaddrs);

            if (temp == NULL)
                std::cout << "NULL\n";

            for (size_t i = 0; (i < this->pages_dma.get()->n_ioaddrs) ; i++) {
                for (size_t j = 0; (j < how_many_in_one); j++) {
                    temp[i*how_many_in_one + j] = ((uint64_t)this->pages_dma.get()->ioaddrs[i]) + j*CL_SIZE_;
                }
            }
            cuda_err_chk(cudaMemcpy(prp1, temp, num_sets_ * num_ways_ * sizeof(uint64_t), cudaMemcpyHostToDevice));

            delete temp;
            prps = false;
        }

        // else if ((ps > this->pages_dma.get()->page_size) && (ps <= (this->pages_dma.get()->page_size * 2))) {
        //     this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
        //     pdt.prp1 = (uint64_t*) this->prp1_buf.get();
        //     this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
        //     pdt.prp2 = (uint64_t*) this->prp2_buf.get();
        //     //uint64_t* temp1 = (uint64_t*) malloc(np * sizeof(uint64_t));
        //     uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
        //     std::memset(temp1, 0, np * sizeof(uint64_t));
        //     //uint64_t* temp2 = (uint64_t*) malloc(np * sizeof(uint64_t));
        //     uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
        //     std::memset(temp2, 0, np * sizeof(uint64_t));
        //     for (size_t i = 0; i < np; i++) {
        //         temp1[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2]);
        //         temp2[i] = ((uint64_t)this->pages_dma.get()->ioaddrs[i*2+1]);
        //     }
        //     cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
        //     cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));

        //     delete temp1;
        //     delete temp2;
        //     pdt.prps = true;
        // }
        // else {
        //     this->prp1_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
        //     pdt.prp1 = (uint64_t*) this->prp1_buf.get();
        //     uint32_t prp_list_size =  ctrl.ctrl->page_size  * np;
        //     this->prp_list_dma = createDma(ctrl.ctrl, NVM_PAGE_ALIGN(prp_list_size, 1UL << 16), cudaDevice);
        //     this->prp2_buf = createBuffer(np * sizeof(uint64_t), cudaDevice);
        //     pdt.prp2 = (uint64_t*) this->prp2_buf.get();
        //     uint64_t* temp1 = new uint64_t[np * sizeof(uint64_t)];
        //     uint64_t* temp2 = new uint64_t[np * sizeof(uint64_t)];
        //     uint64_t* temp3 = new uint64_t[prp_list_size];
        //     std::memset(temp1, 0, np * sizeof(uint64_t));
        //     std::memset(temp2, 0, np * sizeof(uint64_t));
        //     std::memset(temp3, 0, prp_list_size);
        //     uint32_t how_many_in_one = ps /  ctrl.ctrl->page_size ;
        //     for (size_t i = 0; i < np; i++) {
        //         temp1[i] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one]);
        //         temp2[i] = ((uint64_t) this->prp_list_dma.get()->ioaddrs[i]);
        //         for(size_t j = 0; j < (how_many_in_one-1); j++) {
        //             temp3[i*uints_per_page + j] = ((uint64_t) this->pages_dma.get()->ioaddrs[i*how_many_in_one + j + 1]);
        //         }
        //     }

        //     std::cout << "Done creating PRP\n";
        //     cuda_err_chk(cudaMemcpy(pdt.prp1, temp1, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
        //     cuda_err_chk(cudaMemcpy(pdt.prp2, temp2, np * sizeof(uint64_t), cudaMemcpyHostToDevice));
        //     cuda_err_chk(cudaMemcpy(this->prp_list_dma.get()->vaddr, temp3, prp_list_size, cudaMemcpyHostToDevice));

        //     delete temp1;
        //     delete temp2;
        //     delete temp3;
        //     pdt.prps = true;
        // }


        SA_cache_d_t<T> cache_host(d_set_locks, d_way_locks, num_sets_, num_ways_, CL_SIZE_, keys_, set_cnt_, d_ctrls, n_ctrls, n_blocks_per_page, base_addr, prp1, prp2, prps,
        cache_q_head, cache_q_tail, cache_q_lock, cache_extra_reads);

        cuda_err_chk(cudaMemcpy(cache_ptr, &cache_host, sizeof(SA_cache_d_t<T>), cudaMemcpyHostToDevice));
    }

    __host__ 
    SA_cache_d_t<T>* 
    get_ptr(){
        return cache_ptr;
    }



};






#endif // __SET_ASSOCIATIVE_PAGE_CACHE_H__
