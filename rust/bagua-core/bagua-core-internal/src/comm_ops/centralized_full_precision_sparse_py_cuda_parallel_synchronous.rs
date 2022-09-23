use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensor, BaguaTensorRaw, RawBaguaTensor};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::BaguaCommOpChannels;
use crate::kernels;
use std::sync::Arc;

#[derive(Debug)]
pub struct CentralizedFullPrecisionSparsePyCudaParallelSynchronous {
    pub communicator: BaguaCommunicator,
    pub recv_value: BaguaTensor,
    pub recv_index: BaguaTensor,
    pub send_value: BaguaTensor,
    pub other_value: BaguaTensor,
}

impl CommOpTrait for CentralizedFullPrecisionSparsePyCudaParallelSynchronous {
    fn execute_background_communication(
        &self,
        bucket: Arc<BaguaBucket>,
        _comm_op_channels: &BaguaCommOpChannels,
    ) {
        let bucket = bucket.inner.lock();
        let stream_ptr = self.communicator.stream_ptr();
        let mut communication_tensor = bucket.get_communication_tensor(stream_ptr, false, false);
        let mut recv_value_guard = self.recv_value.inner.write();
        let mut recv_value_raw = recv_value_guard.raw.as_mut();
        let mut recv_index_guard = self.recv_index.inner.write();
        let mut recv_index_raw = recv_index_guard.raw.as_mut();
        let mut send_value_guard = self.send_value.inner.write();
        let mut send_value_raw = send_value_guard.raw.as_mut();
        let mut other_value_guard = self.other_value.inner.write();
        let mut other_value_raw = other_value_guard.raw.as_mut();
        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            true,
            true,
            &mut |c, t| {
                tracing::debug!("internode communication started");
                tracing::debug!("start alltoall");
                println!("index tensor raw length: {}, 0.", t.raw.num_elements());

                unsafe {
                    kernels::sparse_extract_parallel_host(
                        other_value_raw.data_ptr() as _,
                        t.raw.data_ptr() as _,
                        t.raw.num_elements() as _,
                        send_value_raw.data_ptr() as _,
                        c.stream_ptr as _,
                    );
                }
                println!("index tensor raw length: {}, 1.", t.raw.num_elements());

                c.allgather(send_value_raw, recv_value_raw);
                c.allgather(&mut t.raw, recv_index_raw);
                println!("index tensor raw length: {}, 2.", t.raw.num_elements());

                unsafe {
                    // method 1:
                    let mut index_ptr = recv_index_raw.data_ptr();
                    let mut value_ptr = recv_value_raw.data_ptr();
                    let mut total_count = other_value_raw.num_elements();
                    for i in 0..c.nranks {
                        if i > 0 {
                            total_count = 0;
                        }
                        kernels::sparse_gather_parallel_host(
                            value_ptr as _,
                            index_ptr as _,
                            t.raw.num_elements() as _,
                            other_value_raw.data_ptr() as _,
                            total_count as _,
                            c.nranks as _,
                            c.stream_ptr as _,
                        );
                        index_ptr += t.raw.num_elements() as u64 * t.raw.dtype().bytes() as u64;
                        value_ptr += send_value_raw.num_elements() as u64 * send_value_raw.dtype().bytes() as u64;
                    }
                    /////////////////////////
                    // method 2:
                    // kernels::sparse_gather_parallel_host(
                    //     recv_value_raw.data_ptr() as _,
                    //     recv_index_raw.data_ptr() as _,
                    //     recv_index_raw.num_elements() as _,
                    //     other_value_raw.data_ptr() as _,
                    //     other_value_raw.num_elements() as _,
                    //     c.nranks as _,
                    //     c.stream_ptr as _,
                    // );
                    //////////////////////////
                }
                println!("index tensor raw length: {}, 3.", t.raw.num_elements());

                other_value_raw.divide_inplace(c.stream_ptr, c.nranks as f32);
                tracing::debug!("internode communication done")
            },
        );
    }
}
