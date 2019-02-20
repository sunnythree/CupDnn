package cupcnn.optimizer;

import java.util.HashMap;
import java.util.List;

import cupcnn.data.Blob;
import cupcnn.optimizer.Optimizer.GMode;
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
	public void updateB(List<Blob> params, List<Blob> gradient) {
		// TODO Auto-generated method stub
		assert params.size()==gradient.size():"params size not equal gradient size";
		if(params.size()==0)return;
		Blob priv = privMap.get(params.get(0));
		if(priv == null) {
			priv = new Blob(params.get(0),false);
			privMap.put(params.get(0), priv);
		}else {
			assert priv.getSize()==params.get(0).getSize():"momentum size error";
		}
		
		float[] privData = priv.getData();
		
		for(int i=0;i<params.size();i++){
			Blob param = params.get(i);
			Blob grad = gradient.get(i);
			float[] paramData = param.getData();
			float[] gradData = grad.getData();
			assert param.getSize()==grad.getSize():"param data size not equal gradient data size";
			for(int j=0;j<param.getSize();j++){
				float V = momentum*privData[j]-lr*gradData[j];
				paramData[j] += V;
				privData[j] = V;
			}
		}
	}
	@Override
	public void updateW(List<Blob> params, List<Blob> gradient) {
		// TODO Auto-generated method stub
		assert params.size()==gradient.size():"params size not equal gradient size";
		if(params.size()==0)return;
		Blob priv = privMap.get(params.get(0));
		if(priv == null) {
			priv = new Blob(params.get(0),false);
			privMap.put(params.get(0), priv);
		}else {
			assert priv.getSize()==params.get(0).getSize():"momentum size error";
		}
		float[] privData = priv.getData();
		for(int i=0;i<params.size();i++){
			Blob param = params.get(i);
			Blob grad = gradient.get(i);
			float[] paramData = param.getData();
			float[] gradData = grad.getData();
			assert param.getSize()==grad.getSize():"param data size not equal gradient data size";
			if(mode==GMode.L2) {
				for(int j=0;j<param.getSize();j++){
					//添加l2衰减
					float V = momentum*privData[j]-lr*lamda*paramData[j]  - lr*gradData[j];
					paramData[j] += V;
					privData[j] = V;
				}
			}else if(mode==GMode.L1){
				for(int j=0;j<param.getSize();j++){
					//添加l1衰减
					float V = 0;
					if(paramData[j]>=0) {
						V = momentum*privData[j] - lr*lamda  - lr*gradData[j];
					}else {
						V = momentum*privData[j] + lr*lamda - lr*gradData[j];
					}
					paramData[j] += V;
					privData[j] = V;
				}				
			}else {
				for(int j=0;j<param.getSize();j++){
					float V = momentum*privData[j]-lr*gradData[j];
					paramData[j] += V;
					privData[j] = V;
				}
			}
		}
	}
}
