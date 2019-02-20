package cupcnn.loss;

import cupcnn.data.Blob;

public class LogLikeHoodLoss extends Loss{

	@Override
	public float loss(Blob label, Blob output) {
		// TODO Auto-generated method stub
		float[] labelData = label.getData();
		float[] outputData = output.getData();
		float loss = 0.0f;
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
		float[] labelData = label.getData();
		float[] outputData = output.getData();
		float[] diffData = diff.getData();
		diff.fillValue(0.0f);
		assert diffData.length == outputData.length:"LogLikeHoodLoss diff --- diffData.length == outputData.length error";
		assert labelData.length == outputData.length:"LogLikeHoodLoss diff --- labelData.length == outputData.length error";
		for(int n=0;n<output.getNumbers();n++){
			for(int os=0;os<output.get3DSize();os++){
				diffData[n*output.get3DSize()+os] -= labelData[n*output.get3DSize()+os]/outputData[n*output.get3DSize()+os];
			}
		}
	}

}
