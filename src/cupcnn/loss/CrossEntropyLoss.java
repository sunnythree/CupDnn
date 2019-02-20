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
		int height = label.getHeight();
		int width = label.getWidth();
		diff.fillValue(0.0f);
		assert diffData.length == outputData.length:"CrossEntropyLoss diff --- diffData.length == outputData.length error";
		assert labelData.length == outputData.length:"CrossEntropyLoss diff --- labelData.length == outputData.length error";
		for(int n=0;n<height;n++){
			for(int os=0;os<output.getWidth();os++){
				diffData[n*width+os] -= (labelData[n*width+os]/outputData[n*width+os]
						-(1-labelData[n*width+os])/(1-outputData[n*width+os]))/(float)height;
			}
		}
	}

}
