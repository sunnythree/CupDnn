package cupdnn.util;

import java.util.concurrent.Callable;

public class Task<V> implements Callable<V> {
	protected int n = 0;
	public Task(int n) {
		this.n = n;
	}
	@Override
    public V call() throws Exception {
       return null;
    }
}
