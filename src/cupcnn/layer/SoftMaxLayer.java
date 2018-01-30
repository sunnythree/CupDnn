package cupcnn.layer;

import cupcnn.Network;
import cupcnn.data.Blob;
import cupcnn.data.BlobParams;

public class SoftMaxLayer extends Layer{

	public SoftMaxLayer(Network network, BlobParams parames) {
		super(network, parames);
		// TODO Auto-generated constructor stub
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return "SoftMaxLayer";
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		Blob input = mNetwork.getDatas().get(id-1);
		Blob output = mNetwork.getDatas().get(id);
		double[] inputData = input.getData();
		double[] outputData = output.getData();
		assert input.getSize()==output.getSize():"SoftMax forward---- input.getSize()==output.getSize() error";
	
		for(int n=0;n<input.getNumbers();n++){
			double sum = 0.0;
//			double max = 0.0;
			
//			//查找最大值
//			for(int is=0;is<input.get3DSize();is++){
//				max = Math.max(max, inputData[n*input.get3DSize()+is]);
//			}
//			//求和
//			for(int is=0;is<input.get3DSize();is++){
//				outputData[n*input.get3DSize()+is] = Math.exp(inputData[n*input.get3DSize()+is]-max);
//				sum += outputData[n*input.get3DSize()+is];
//			}
//			//每一项除以sum
//			for(int os=0;os<output.get3DSize();os++){
//				outputData[n*output.get3DSize()+os] = outputData[n*output.get3DSize()+os]/sum;
//			}
			
			//求和
			for(int is=0;is<input.get3DSize();is++){
				sum += Math.exp(inputData[n*input.get3DSize()+is]);
			}
			//每一项除以sum
			for(int os=0;os<output.get3DSize();os++){
				outputData[n*output.get3DSize()+os] = Math.exp(inputData[n*output.get3DSize()+os])/sum;
			}
		}
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		Blob inputDiff = mNetwork.getDiffs().get(id);
		Blob outputDiff = mNetwork.getDiffs().get(id-1);
		Blob output = mNetwork.getDatas().get(id);
		double[] inputDiffData = inputDiff.getData();
		double[] outputDiffData = outputDiff.getData();
		double[] outputData = output.getData();
		assert inputDiff.getSize()==outputDiff.getSize():"SoftMax backward---- inputDiff.getSize()==outputDiff.getSize() error";
		
		//先求softmax函数的偏导数
		outputDiff.fillValue(0);
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int ods=0;ods<outputDiff.get3DSize();ods++){
				for(int ids=0;ids<inputDiff.get3DSize();ids++){
					if(ids==ods){
						outputDiffData[n*output.get3DSize()+ods] += outputData[n*output.get3DSize()+ods]*(1.0-outputData[n*output.get3DSize()+ods])
								*inputDiffData[n*output.get3DSize()+ids];
					}else{
						outputDiffData[n*output.get3DSize()+ods] -= outputData[n*output.get3DSize()+ods]*outputData[n*output.get3DSize()+ids]
								*inputDiffData[n*output.get3DSize()+ids];
					}
				}
			}
		}
	}

}
