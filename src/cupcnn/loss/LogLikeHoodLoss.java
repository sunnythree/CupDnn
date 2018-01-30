package cupcnn.loss;

import cupcnn.data.Blob;

public class LogLikeHoodLoss extends Loss{

	@Override
	public double loss(Blob label, Blob output) {
		// TODO Auto-generated method stub
		double[] labelData = label.getData();
		double[] outputData = output.getData();
		double loss = 0.0;
		for(int n=0;n<label.getNumbers();n++){
			for(int i=0;i<label.get3DSize();i++){
				loss -= labelData[n*output.get3DSize()+i]*Math.log(outputData[n*output.get3DSize()+i]);
			}
		}
		loss = loss/label.getNumbers();
		return loss;
	}

	@Override
	public void diff(Blob label, Blob output, Blob diff) {
		// TODO Auto-generated method stub
		double[] labelData = label.getData();
		double[] outputData = output.getData();
		double[] diffData = diff.getData();
		diff.fillValue(0.0);
		assert diffData.length == outputData.length:"LogLikeHoodLoss diff --- diffData.length == outputData.length error";
		assert labelData.length == outputData.length:"LogLikeHoodLoss diff --- labelData.length == outputData.length error";
		for(int n=0;n<output.getNumbers();n++){
			for(int os=0;os<output.get3DSize();os++){
				diffData[n*output.get3DSize()+os] -= labelData[n*output.get3DSize()+os]/outputData[n*output.get3DSize()+os];
			}
		}
	}

}
