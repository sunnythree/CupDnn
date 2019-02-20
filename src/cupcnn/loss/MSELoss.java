package cupcnn.loss;

import cupcnn.data.Blob;

public class MSELoss extends Loss{

	@Override
	public float loss(Blob label, Blob output) {
		// TODO Auto-generated method stub
		float[] labelData = label.getData();
		float[] outputData = output.getData();
		float loss = 0.0f;
	    for (int i = 0; i < label.getSize(); ++i) {
	    	loss += (labelData[i] - outputData[i]) * (labelData[i] - outputData[i]);
	    } 

	    return loss /label.getHeight();
	}

	@Override
	public void diff(Blob label, Blob output, Blob diff) {
		// TODO Auto-generated method stub
		float[] labelData = label.getData();
		float[] outputData = output.getData();
		float[] diffData = diff.getData();
		float factor = 2/(float)label.getWidth();
		diff.fillValue(0.0f);
		assert diffData.length == outputData.length:"MSELoss diff --- diffData.length == outputData.length error";
		assert labelData.length == outputData.length:"MSEEntropyLoss diff --- labelData.length == outputData.length error";
		for(int n=0;n<output.getHeight();n++){
			for(int os=0;os<output.getWidth();os++){
				diffData[n*diff.getWidth()+os] += factor*(outputData[n*diff.getWidth()+os]-labelData[n*diff.getWidth()+os]);
			}
		}
	}

}
