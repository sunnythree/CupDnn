package cupcnn.layer;

import cupcnn.Network;
import cupcnn.data.Blob;
import cupcnn.data.BlobParams;

public class PoolMeanLayer extends Layer{
	private BlobParams kernelParams;
	private int kernelHeightStride = 0;
	private int kernelWidthStride = 0;
	
	public PoolMeanLayer(Network network, BlobParams parames,BlobParams kernelParams,int kernelHeightStride,int kernelWidthStride) {
		super(network, parames);
		// TODO Auto-generated constructor stub
		this.kernelParams = kernelParams;
		this.kernelHeightStride = kernelHeightStride;
		this.kernelWidthStride = kernelWidthStride;
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return "PoolMeanLayer";
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
		double [] outputData = output.getData();
		double [] inputData = input.getData();
		for(int n=0;n<output.getNumbers();n++){
			for(int c=0;c<output.getChannels();c++){
				for(int h=0;h<output.getHeight();h++){
					for(int w=0;w<output.getWidth();w++){
						int inStartX = w*kernelWidthStride;
						int inStartY = h*kernelHeightStride;
						double sum = 0;
						for(int kh=0;kh<kernelParams.getHeight();kh++){
							for(int kw=0;kw<kernelParams.getWidth();kw++){
								int curIndex = input.getIndexByParams(n, c, inStartY+kh, inStartX+kw);
								sum += inputData[curIndex];
							}
						}
						outputData[output.getIndexByParams(n, c, h, w)] = sum/(kernelParams.getHeight()*kernelParams.getWidth());
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
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int c=0;c<inputDiff.getChannels();c++){
				for(int h=0;h<inputDiff.getHeight();h++){
					for(int w=0;w<inputDiff.getWidth();w++){
						int inStartX = w*kernelWidthStride;
						int inStartY = h*kernelHeightStride;
						for(int kh=0;kh<kernelParams.getHeight();kh++){
							for(int kw=0;kw<kernelParams.getWidth();kw++){
								int curIndex = outputDiff.getIndexByParams(n, c, inStartY+kh, inStartX+kw);
								outputDiffData[curIndex] = inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
							}
						}
						
					}
				}
			}
		}	
	}

}
