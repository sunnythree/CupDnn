package cupcnn.optimizer;
import java.util.List;

import cupcnn.data.Blob;

public abstract class Optimizer {
	protected double lr = 0.0;
	public Optimizer(double lr){
		this.lr = lr;
	}
	public abstract void update(List<Blob> params,List<Blob> gradient);
	public void setLr(double lr){
		this.lr = lr;
	}
	public double getLr(){
		return this.lr;
	}
}
