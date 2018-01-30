package cupcnn.optimizer;

import java.util.List;

import cupcnn.data.Blob;

public class SGDOptimizer extends Optimizer {
	
	public SGDOptimizer(double lr){
		super(lr);
	}

	@Override
	public void update(List<Blob> params, List<Blob> gradient) {
		// TODO Auto-generated method stub
		assert params.size()==gradient.size():"params size not equal gradient size";
		for(int i=0;i<params.size();i++){
			Blob param = params.get(i);
			Blob grad = gradient.get(i);
			double[] paramData = param.getData();
			double[] gradData = grad.getData();
			assert param.getSize()==grad.getSize():"param data size not equal gradient data size";
			for(int j=0;j<param.getSize();j++){
				paramData[j] -= lr*gradData[j];
			}
		}
	}

}
