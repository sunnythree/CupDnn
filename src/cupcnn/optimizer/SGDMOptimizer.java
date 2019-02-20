package cupcnn.optimizer;

import java.util.HashMap;
import java.util.List;

import cupcnn.data.Blob;
import cupcnn.optimizer.Optimizer.GMode;
/*
 * SGD with momentum
 */
public class SGDMOptimizer extends Optimizer {
	
	private double momentum = 0.9f;
	private HashMap<Blob,Blob> privMap = new HashMap();
	
	public SGDMOptimizer(double lr,double mententum){
		super(lr);
		this.momentum = mententum;
	}

	
	public SGDMOptimizer(double lr,Optimizer.GMode mode,double lamda,double mententum){
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
		
		double[] privData = priv.getData();
		
		for(int i=0;i<params.size();i++){
			Blob param = params.get(i);
			Blob grad = gradient.get(i);
			double[] paramData = param.getData();
			double[] gradData = grad.getData();
			assert param.getSize()==grad.getSize():"param data size not equal gradient data size";
			for(int j=0;j<param.getSize();j++){
				double V = momentum*privData[j]-lr*gradData[j];
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
		double[] privData = priv.getData();
		for(int i=0;i<params.size();i++){
			Blob param = params.get(i);
			Blob grad = gradient.get(i);
			double[] paramData = param.getData();
			double[] gradData = grad.getData();
			assert param.getSize()==grad.getSize():"param data size not equal gradient data size";
			if(mode==GMode.L2) {
				for(int j=0;j<param.getSize();j++){
					//Ìí¼Ól2Ë¥¼õ
					double V = momentum*privData[j]-lr*lamda*paramData[j]  - lr*gradData[j];
					paramData[j] += V;
					privData[j] = V;
				}
			}else if(mode==GMode.L1){
				for(int j=0;j<param.getSize();j++){
					//Ìí¼Ól1Ë¥¼õ
					double V = 0;
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
					double V = momentum*privData[j]-lr*gradData[j];
					paramData[j] += V;
					privData[j] = V;
				}
			}
		}
	}
}
