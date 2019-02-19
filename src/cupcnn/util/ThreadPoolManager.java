package cupcnn.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import cupcnn.Network;

public class ThreadPoolManager {
	private static ThreadPoolManager instance;
	private static int threadNum = 4;
	public static ThreadPoolManager getInstance(Network network) {
		synchronized(ThreadPoolManager.class) {
			if(instance==null) {
				instance = new ThreadPoolManager(network.getThreadNum());
			}else if(network.getThreadNum()!=threadNum){
				//线程数不相等则重新创建
				instance = new ThreadPoolManager(network.getThreadNum());
			}
		}
		return instance;
	}
	
	//1、配置线程池
    private ExecutorService threadPool;
	private ThreadPoolManager(int num) {
		if(num<1) {
			threadNum = 4;
			threadPool = Executors.newFixedThreadPool(4);
		}else {
			threadNum = num;
			threadPool = Executors.newFixedThreadPool(num);
		}
	}
	public void dispatchTask(Vector<Callable> workers) {
		//接收多线程响应结果
		Vector<Future> results = new Vector<Future>();
        for(Callable c: workers) {
        	Future f = threadPool.submit(c);
        	results.add(f);
        }
        for(int i=0;i<results.size();i++) {
        	try {
				results.get(i).get();
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (ExecutionException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
        }
	}
}
