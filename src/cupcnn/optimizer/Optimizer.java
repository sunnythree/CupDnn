package cupcnn.optimizer;
import java.util.List;

import cupcnn.data.Blob;

public abstract class Optimizer {
	protected double lr = 0.0;
	protected double lamda = 0.0;
	public static enum GMode{
		NONE,
		L1,
		L2
	}
	GMode mode;
	public Optimizer(double lr){
		this.lr = lr;
		this.mode = GMode.NONE;
	}
	
	
	public Optimizer(double lr,GMode mode,double lamda){
		this.lr = lr;
		this.lamda = lamda;
		this.mode = mode;
	}
	public abstract void updateW(List<Blob> params,List<Blob> gradient);
	public abstract void updateB(List<Blob> params,List<Blob> gradient);
	public void setLr(double lr){
		this.lr = lr;
	}
	public double getLr(){
		return this.lr;
	}

}
