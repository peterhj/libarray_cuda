extern crate array_cuda;
extern crate scoped_threadpool;

use array_cuda::context::{DeviceContext};
use array_cuda::device_comm::{DeviceReduceWorker};
use array_cuda::device_ext::*;
use array_cuda::device_memory::{DeviceBuffer, SyncDeviceBuffer};
use scoped_threadpool::{Pool};

fn main() {
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
      reduce_bufs.push(SyncDeviceBuffer::<i32>::new(1024, &ctx));
    }
    ctxs.clear();
    //for (tid, ctx) in ctxs.into_iter() {
    for tid in (0 .. num_threads) {
      let reduce = DeviceReduceWorker::<i32>::new(tid, num_threads, &reduce_bufs);
      scope.execute(move || {
        println!("thread {}: hello world!", tid);
        let context = DeviceContext::new(tid);
        let ctx = context.as_ref();
        let mut reduce = reduce;
        let mut input = DeviceBuffer::new(1024, &ctx);
        {
          let mut input = input.borrow_mut(&ctx);
          match tid {
            0 => input.set_constant(1000_0000),
            1 => input.set_constant(0200_0000),
            2 => input.set_constant(0030_0000),
            3 => input.set_constant(0004_0000),
            4 => input.set_constant(0000_5000),
            5 => input.set_constant(0000_0600),
            6 => input.set_constant(0000_0070),
            7 => input.set_constant(0000_0008),
            _ => unreachable!(),
          }
        }
        let mut input = input.borrow(&ctx);
        reduce.process(&mut input, &ctx);
        println!("thread {}: goodbye world!", tid);
      });
    }
    scope.join_all();
  });
  println!("finished");
}
