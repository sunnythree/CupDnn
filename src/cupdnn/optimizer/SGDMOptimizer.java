package cupdnn.optimizer;

import java.util.HashMap;
import java.util.List;

import cupdnn.data.Blob;
import cupdnn.optimizer.Optimizer.GMode;
/*
 * SGD with momentum
 */
public class SGDMOptimizer extends Optimizer {
	
	private float momentum = 0.9f;
	private HashMap<Blob,Blob> privMap = new HashMap();
	
	public SGDMOptimizer(float lr,float mententum){
		super(lr);
		this.momentum = mententum;
	}

	/*
	 * lamda是衰减权重，是一个很小的数字
	 * */
	
	public SGDMOptimizer(float lr,Optimizer.GMode mode,float lamda,float mententum){
		super(lr,mode,lamda);
		this.momentum = mententum;
	}

	@Override
	public void updateB(Blob b, Blob gradient) {
		// TODO Auto-generated method stub
		Blob priv = privMap.get(b);
		if(priv == null) {
			priv = new Blob(b,false);
			privMap.put(b, priv);
		}
		
		float[] privData = priv.getData();
		float[] bData = b.getData();
		float[] gradData = gradient.getData();
		for(int j=0;j<b.getSize();j++){
			float V = momentum*privData[j]-lr*gradData[j];
			bData[j] += V;
			privData[j] = V;
		}
	}
	@Override
	public void updateW(Blob w, Blob gradient) {
		// TODO Auto-generated method stub
		Blob priv = privMap.get(w);
		if(priv == null) {
			priv = new Blob(w,false);
			privMap.put(w, priv);
		}
		float[] privData = priv.getData();
		float[] wData = w.getData();
		float[] gradData = gradient.getData();
		if(mode==GMode.L2) {
			for(int j=0;j<w.getSize();j++){
				//添加l2衰减
				float V = momentum*privData[j]-lr*lamda*wData[j]  - lr*gradData[j];
				wData[j] += V;
				privData[j] = V;
			}
		}else if(mode==GMode.L1){
			for(int j=0;j<w.getSize();j++){
				//添加l1衰减
				float V = 0;
				if(wData[j]>=0) {
					V = momentum*privData[j] - lr*lamda  - lr*gradData[j];
				}else {
					V = momentum*privData[j] + lr*lamda - lr*gradData[j];
				}
				wData[j] += V;
				privData[j] = V;
			}				
		}else {
			for(int j=0;j<w.getSize();j++){
				float V = momentum*privData[j]-lr*gradData[j];
				wData[j] += V;
				privData[j] = V;
			}
		}
	}
}
