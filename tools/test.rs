extern crate array_cuda;
extern crate scoped_threadpool;
extern crate time;

use array_cuda::context::{DeviceContext};
use array_cuda::device_comm::{DeviceReduceWorker, DeviceAllReduceWorker};
use array_cuda::device_ext::*;
use array_cuda::device_memory::{DeviceBuffer, RawDeviceBuffer, SyncDeviceBuffer};
use scoped_threadpool::{Pool};

use std::sync::{Arc, Barrier};
use time::{PreciseTime};

fn main() {
  let n = 6 * 1024 * 1024;
  let num_threads = 8;
  let mut pool = Pool::new(num_threads as u32);
  pool.scoped(|scope| {
    let mut ctxs = vec![];
    for tid in (0 .. num_threads) {
      ctxs.push(DeviceContext::new(tid));
    }
    let mut reduce_bufs = vec![];
    for i in (0 .. 2 * num_threads) {
      let ctx = ctxs[i % num_threads].as_ref();
      //reduce_bufs.push(SyncDeviceBuffer::<f32>::new(n, &ctx));
      reduce_bufs.push(Arc::new(RawDeviceBuffer::<f32>::new(n, &ctx)));
    }
    ctxs.clear();

    let barrier = Arc::new(Barrier::new(num_threads));
    for tid in (0 .. num_threads) {
      let barrier = barrier.clone();
      //let mut reduce = DeviceReduceWorker::<f32>::new(tid, num_threads, barrier, &reduce_bufs);
      let mut allreduce = DeviceAllReduceWorker::<f32>::new(tid, num_threads, barrier, &reduce_bufs);
      scope.execute(move || {
        println!("thread {}: hello world!", tid);
        let context = DeviceContext::new(tid);
        let ctx = context.as_ref();
        //let mut reduce = reduce;
        let mut input = DeviceBuffer::new(n, &ctx);
        {
          let mut input = input.borrow_mut(&ctx);
          input.set_constant(1.0);
          /*match tid {
            0 => input.set_constant(1000_0000),
            1 => input.set_constant(0200_0000),
            2 => input.set_constant(0030_0000),
            3 => input.set_constant(0004_0000),
            4 => input.set_constant(0000_5000),
            5 => input.set_constant(0000_0600),
            6 => input.set_constant(0000_0070),
            7 => input.set_constant(0000_0008),
            _ => unreachable!(),
          }*/
        }
        let mut input = input.borrow(&ctx);
        let ntrials = 100;
        let start_time = PreciseTime::now();
        for _ in (0 .. ntrials) {
          //reduce.process(&mut input, &ctx);
          allreduce.process(&mut input, &ctx);
        }
        let stop_time = PreciseTime::now();
        let elapsed = start_time.to(stop_time);
        let elapsed_ns = elapsed.num_nanoseconds().unwrap();
        println!("thread {}: goodbye world: avg elapsed {} ns", tid, elapsed_ns / ntrials);
      });
    }
    scope.join_all();
  });
  println!("finished");
}
