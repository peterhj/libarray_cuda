extern crate array_cuda;
extern crate scoped_threadpool;
extern crate time;

use array_cuda::context::{DeviceContext};
use array_cuda::device_comm::{DeviceAllReduceSharedData, DeviceAllReduceWorker};
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
    let allreduce_shared = Arc::new(DeviceAllReduceSharedData::<f32>::new(num_threads, n, &ctxs));
    ctxs.clear();

    for tid in (0 .. num_threads) {
      let allreduce_shared = allreduce_shared.clone();
      scope.execute(move || {
        println!("thread {}: hello world!", tid);
        let context = DeviceContext::new(tid);
        let ctx = context.as_ref();
        let mut allreduce = DeviceAllReduceWorker::<f32>::new(tid, allreduce_shared, &ctx);
        let mut input = DeviceBuffer::new(n, &ctx);
        {
          let mut input = input.borrow_mut(&ctx);
          input.set_constant(1.0);
        }
        let mut input = input.borrow(&ctx);
        let ntrials = 100;
        let start_time = PreciseTime::now();
        for _ in (0 .. ntrials) {
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
