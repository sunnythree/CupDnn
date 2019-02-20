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
	public void updateB(List<Blob> params, List<Blob> gradient) {
		// TODO Auto-generated method stub
		assert params.size()==gradient.size():"params size not equal gradient size";
		for(int i=0;i<params.size();i++){
			Blob param = params.get(i);
			Blob grad = gradient.get(i);
			float[] paramData = param.getData();
			float[] gradData = grad.getData();
			assert param.getSize()==grad.getSize():"param data size not equal gradient data size";
			for(int j=0;j<param.getSize();j++){
				paramData[j] -= lr*gradData[j];
			}
		}
	}
	@Override
	public void updateW(List<Blob> params, List<Blob> gradient) {
		// TODO Auto-generated method stub
		assert params.size()==gradient.size():"params size not equal gradient size";
		for(int i=0;i<params.size();i++){
			Blob param = params.get(i);
			Blob grad = gradient.get(i);
			float[] paramData = param.getData();
			float[] gradData = grad.getData();
			assert param.getSize()==grad.getSize():"param data size not equal gradient data size";
			if(mode==GMode.L2) {
				for(int j=0;j<param.getSize();j++){
					//Ìí¼Ól2Ë¥¼õ
					paramData[j] = (1.0f-lr*lamda)*paramData[j]  - lr*gradData[j];
				}
			}else if(mode==GMode.L1){
				for(int j=0;j<param.getSize();j++){
					//Ìí¼Ól1Ë¥¼õ
					if(paramData[j]>=0) {
						paramData[j] = paramData[j] - lr*lamda - lr*gradData[j];
					}else {
						paramData[j] = paramData[j] + lr*lamda - lr*gradData[j];
					}
				}				
			}else {
				for(int j=0;j<param.getSize();j++){
					paramData[j] -= lr*gradData[j];
				}
			}
		}
	}
}
