package cupcnn.layer;

import cupcnn.Network;
import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.util.MathFunctions;

public class ConvolutionLayer extends Layer{

	private Blob kernel;
	private Blob bias;
	private Blob kernelGradient;
	private Blob biasGradient;
	private Blob z;
	private BlobParams kernelParams;

	public ConvolutionLayer(Network network, BlobParams layerParsms,BlobParams kernelParams) {
		// TODO Auto-generated constructor stub
		super(network, layerParsms);
		this.kernelParams = kernelParams;
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return "ConvolutionLayer";
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		Blob output = mNetwork.getDatas().get(id);
		//layerParams.getHeight()表示该层需要提取的特征数量
		kernel = new Blob(kernelParams.getNumbers(),kernelParams.getChannels(),kernelParams.getHeight(),kernelParams.getHeight());
		kernelGradient = new Blob(kernel.getNumbers(),kernel.getChannels(),kernel.getHeight(),kernel.getWidth());
		bias = new Blob(kernelParams.getNumbers(),kernelParams.getChannels(),1,1);
		biasGradient = new Blob(bias.getNumbers(),bias.getChannels(),bias.getHeight(),bias.getWidth());
		z = new Blob(output.getNumbers(),output.getChannels(),output.getHeight(),output.getWidth());
		
		//init params
		MathFunctions.gaussianInitData(kernel.getData());
		MathFunctions.constantInitData(bias.getData(), 0.1);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		Blob input = mNetwork.getDatas().get(id-1);
		Blob output = mNetwork.getDatas().get(id);
		double [] outputData = output.getData();
		double [] zData = z.getData();
		//卷积后的结果存贮在z中
		z.fillValue(0);
		MathFunctions.convolutionBlobSame(input, kernel, bias, z);
		//激活函数
		if(activationFunc!=null){
			for(int n=0;n<output.getNumbers();n++){
				for(int c=0;c<output.getChannels();c++){
					for(int h=0;h<output.getHeight();h++){
						for(int w=0;w<output.getWidth();w++){
							outputData[output.getIndexByParams(n, c, h, w)] = activationFunc.active(zData[z.getIndexByParams(n, c, h, w)]);
						}
					}
				}
			}
		}
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		Blob input = mNetwork.getDatas().get(id-1);
		Blob inputDiff = mNetwork.getDiffs().get(id);
		Blob outputDiff = mNetwork.getDiffs().get(id-1);
		double[] inputDiffData = inputDiff.getData();
		double[] zData = z.getData();
		double[] kernelGradientData = kernelGradient.getData();
		double[] inputData = input.getData();
		double[] biasGradientData = biasGradient.getData();
		
		//先乘激活函数的导数,得到该层的误差
		if(activationFunc!=null){
			for(int n=0;n<inputDiff.getNumbers();n++){
				for(int c=0;c<inputDiff.getChannels();c++){
					for(int h=0;h<inputDiff.getHeight();h++){
						for(int w=0;w<inputDiff.getWidth();w++){
							inputDiffData[inputDiff.getIndexByParams(n, c, h, w)] *= activationFunc.diffActive(zData[z.getIndexByParams(n, c, h, w)]);
						}
					}
				}
			}
		}
		
		//然后更新参数
		//更新kernel
		kernelGradient.fillValue(0);
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int c=0;c<inputDiff.getChannels();c++){
				int inputChannelIndex = c/(inputDiff.getChannels()/input.getChannels());
				for(int h=0;h<inputDiff.getHeight();h++){
					for(int w=0;w<inputDiff.getWidth();w++){
						//先定位到输出的位置
						//然后遍历kernel,通过kernel定位输入的位置
						//然后将输入乘以kernel
						int inStartX = w - kernelGradient.getWidth()/2;
						int inStartY = h - kernelGradient.getHeight()/2;
						//和卷积核乘加
			
						for(int kh=0;kh<kernelGradient.getHeight();kh++){
							for(int kw=0;kw<kernelGradient.getWidth();kw++){
								int inY = inStartY + kh;
								int inX = inStartX + kw;
								if (inY >= 0 && inY < input.getHeight() && inX >= 0 && inX < input.getWidth()){
									kernelGradientData[kernelGradient.getIndexByParams(0,  c, kh, kw)] += inputData[input.getIndexByParams(n,inputChannelIndex , inY, inX)]
											*inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
								}
							}
						}
					}
				}
			}
		}
		//平均
		MathFunctions.dataDivConstant(kernelGradientData, inputDiff.getNumbers());
		
		//更新bias
		biasGradient.fillValue(0);
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int c=0;c<inputDiff.getChannels();c++){
				for(int h=0;h<inputDiff.getHeight();h++){
					for(int w=0;w<inputDiff.getWidth();w++){
						biasGradientData[bias.getIndexByParams(0, c, 0, 0)] += inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
					}
				}
			}
		}
		//平均
		MathFunctions.dataDivConstant(biasGradientData, inputDiff.getNumbers());
		
		if(id<=1)return;
		//先把kernel旋转180度
		Blob kernelRoate180 = MathFunctions.rotate180Blob(kernel);
		//然后再做卷积
		MathFunctions.convolutionBlobSame(inputDiff, kernelRoate180, outputDiff);	
		
		paramsList.clear();
		paramsList.add(kernel);
		paramsList.add(bias);
		
		gradientList.clear();
		gradientList.add(kernelGradient);
		gradientList.add(biasGradient);
	}
}
