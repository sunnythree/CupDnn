package cupcnn.layer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupcnn.Network;
import cupcnn.data.Blob;
import cupcnn.data.BlobParams;

public class PoolMaxLayer extends Layer{
	public static final String TYPE = "PoolMaxLayer";
	
	private Blob maxIndex;
	private BlobParams kernelParams;
	private int kernelHeightStride = 0;
	private int kernelWidthStride = 0;
	public PoolMaxLayer(Network network, BlobParams layerParames,BlobParams kernelParams,int kernelHeightStride,int kernelWidthStride) {
		// TODO Auto-generated constructor stub
		super(network, layerParames);
		this.kernelParams = kernelParams;
		this.kernelHeightStride = kernelHeightStride;
		this.kernelWidthStride = kernelWidthStride;
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return TYPE;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		maxIndex = new Blob(layerParams.getNumbers(),layerParams.getChannels(),layerParams.getHeight(),layerParams.getWidth());

	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		
		Blob input = mNetwork.getDatas().get(id-1);
		Blob output = mNetwork.getDatas().get(id);
		double [] outputData = output.getData();
		double [] inputData = input.getData();
		double [] maxIndexData = maxIndex.getData();
		
		for(int n=0;n<output.getNumbers();n++){
			for(int c=0;c<output.getChannels();c++){
				for(int h=0;h<output.getHeight();h++){
					for(int w=0;w<output.getWidth();w++){
						int inStartX = w*kernelWidthStride;
						int inStartY = h*kernelHeightStride;
						double localMaxVlue = 0;
						int localMaxIndex = 0;
						for(int kh=0;kh<kernelParams.getHeight();kh++){
							for(int kw=0;kw<kernelParams.getWidth();kw++){
								int curIndex = input.getIndexByParams(n, c, inStartY+kh, inStartX+kw);
								if(inputData[curIndex]>localMaxVlue){
									localMaxVlue = inputData[curIndex];
									localMaxIndex = kh*kernelParams.getWidth()+kw;
								}
							}
						}
						maxIndexData[maxIndex.getIndexByParams(n, c, h, w)] = localMaxIndex;
						outputData[output.getIndexByParams(n, c, h, w)] = localMaxVlue;
					}
				}
			}
		}	
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		Blob inputDiff = mNetwork.getDiffs().get(id);
		Blob outputDiff = mNetwork.getDiffs().get(id-1);
		double[] inputDiffData = inputDiff.getData();
		double[] outputDiffData = outputDiff.getData();
		double [] maxIndexData = maxIndex.getData();
		
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int c=0;c<inputDiff.getChannels();c++){
				for(int h=0;h<inputDiff.getHeight();h++){
					for(int w=0;w<inputDiff.getWidth();w++){
						int inStartX = w*kernelWidthStride;
						int inStartY = h*kernelHeightStride;
						int iY = (int)maxIndexData[maxIndex.getIndexByParams(n, c, h, w)]/kernelParams.getWidth();
						int iX = (int)maxIndexData[maxIndex.getIndexByParams(n, c, h, w)]%kernelParams.getWidth();

						outputDiffData[outputDiff.getIndexByParams(n, c, inStartY+iY, inStartX+iX)] = inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
					}
				}
			}
		}	
	}

	@Override
	public void saveModel(ObjectOutputStream out) {
		// TODO Auto-generated method stub
		try {
			out.writeUTF(getType());
			//保存的时候，batch也就是layerParams的number总是1，因为predict的时候，因为真正使用的时候，这个batch一般都是1
			layerParams.setNumbers(1);
			out.writeObject(layerParams);
			out.writeObject(kernelParams);
			out.writeInt(kernelHeightStride);
			out.writeInt(kernelWidthStride);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	@Override
	public void loadModel(ObjectInputStream in) {
		// TODO Auto-generated method stub
		//do nothing
	}

}
