use crate::comm_ops::CommOpTrait;
use crate::communicators::BaguaCommunicator;
use crate::datatypes::{BaguaBucket, BaguaReductionOp, BaguaTensor, BaguaTensorRaw, RawBaguaTensor};
use crate::resource_pool::CUDA_DEVICE_MEMORY_POOL;
use crate::BaguaCommOpChannels;
use crate::kernels;
use std::sync::Arc;

#[derive(Debug)]
pub struct CentralizedFullPrecisionSparsePySynchronous {
    pub communicator: BaguaCommunicator,
    pub recv_value: BaguaTensor,
    pub recv_index: BaguaTensor,
    pub send_value: BaguaTensor,
}

impl CommOpTrait for CentralizedFullPrecisionSparsePySynchronous {
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
        self.communicator.execute_communication(
            &mut communication_tensor,
            false,
            true,
            true,
            &mut |c, t| {
                tracing::debug!("internode communication started");
                tracing::debug!("start alltoall");

                c.allgather(send_value_raw, recv_value_raw);
                c.allgather(&mut t.raw, recv_index_raw);

                tracing::debug!("internode communication done")
            },
        );
    }
}
