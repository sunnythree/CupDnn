package cupcnn.loss;

import cupcnn.data.Blob;

public class CrossEntropyLoss extends Loss {

	
	@Override
	public float loss(Blob label, Blob output) {
		// TODO Auto-generated method stub
		float[] labelData = label.getData();
		float[] outputData = output.getData();
		float loss = 0.0f;
		for(int i=0;i<label.getSize();i++){
			loss += labelData[i]*Math.log(outputData[i])+(1-labelData[i])*Math.log(1-outputData[i]);
		}
		loss = -loss/label.getHeight();
		return loss;
	}

	@Override
	public void diff(Blob label, Blob output, Blob diff) {
		// TODO Auto-generated method stub
		float[] labelData = label.getData();
		float[] outputData = output.getData();
		float[] diffData = diff.getData();
		diff.fillValue(0.0f);
		assert diffData.length == outputData.length:"CrossEntropyLoss diff --- diffData.length == outputData.length error";
		assert labelData.length == outputData.length:"CrossEntropyLoss diff --- labelData.length == outputData.length error";
		for(int n=0;n<output.getHeight();n++){
			for(int os=0;os<output.getWidth();os++){
				diffData[n*diff.getWidth()+os] -= (labelData[n*label.getWidth()+os]/outputData[n*output.getWidth()+os]
						-(1-labelData[n*label.getWidth()+os])/(1-outputData[n*output.getWidth()+os]));
			}
		}
	}

}
