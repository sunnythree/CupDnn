package cupcnn.optimizer;

import java.util.List;

import cupcnn.data.Blob;


/*
 * SGD without momentum
 */

public class SGDOptimizer extends Optimizer {
	
	public SGDOptimizer(float lr){
		super(lr);
	}

	
	public SGDOptimizer(float lr,Optimizer.GMode mode,float lamda){
		super(lr,mode,lamda);
	}

	@Override
	public void updateB(Blob b,Blob gradient) {
		// TODO Auto-generated method stub
		float[] bData = b.getData();
		float[] gradData = gradient.getData();
		for(int j=0;j<b.getSize();j++){
			bData[j] -= lr*gradData[j];
		}
	}
	@Override
	public void updateW(Blob w, Blob gradient) {
		// TODO Auto-generated method stub
			float[] wData = w.getData();
			float[] gradData = gradient.getData();
			if(mode==GMode.L2) {
				for(int j=0;j<w.getSize();j++){
					//Ìí¼Ól2Ë¥¼õ
					wData[j] = (1.0f-lr*lamda)*wData[j]  - lr*gradData[j];
				}
			}else if(mode==GMode.L1){
				for(int j=0;j<w.getSize();j++){
					//Ìí¼Ól1Ë¥¼õ
					if(wData[j]>=0) {
						wData[j] = wData[j] - lr*lamda - lr*gradData[j];
					}else {
						wData[j] = wData[j] + lr*lamda - lr*gradData[j];
					}
				}				
			}else {
				for(int j=0;j<w.getSize();j++){
					wData[j] -= lr*gradData[j];
				}
			}
	}
}
