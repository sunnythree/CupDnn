package cupcnn.layer;



import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupcnn.Network;
import cupcnn.active.ReluActivationFunc;
import cupcnn.active.SigmodActivationFunc;
import cupcnn.active.TanhActivationFunc;
import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.util.MathFunctions;

public class ConvolutionLayer extends Layer{
	public static final String TYPE = "ConvolutionLayer";
	private Blob kernel;
	private Blob bias;
	private transient Blob kernelGradient;
	private transient Blob biasGradient;
	private transient Blob z;
	private BlobParams kernelParams;

	public ConvolutionLayer(Network network, BlobParams layerParsms,BlobParams kernelParams) {
		// TODO Auto-generated constructor stub
		super(network, layerParsms);
		this.kernelParams = kernelParams;
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return TYPE;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		Blob output = mNetwork.getDatas().get(id);
		//layerParams.getHeight()表示该层需要提取的特征数量
		if(kernel ==null && bias == null){
			kernel = new Blob(kernelParams.getNumbers(),kernelParams.getChannels(),kernelParams.getHeight(),kernelParams.getHeight());
			bias = new Blob(kernelParams.getNumbers(),kernelParams.getChannels(),1,1);
			//init params
			MathFunctions.gaussianInitData(kernel.getData());
			MathFunctions.constantInitData(bias.getData(), 0.1);
		}
		assert kernel != null && bias != null :"ConvolutionLayer prepare----- kernel is null or bias is null error";
		z = new Blob(output.getNumbers(),output.getChannels(),output.getHeight(),output.getWidth());
		kernelGradient = new Blob(kernel.getNumbers(),kernel.getChannels(),kernel.getHeight(),kernel.getWidth());
		biasGradient = new Blob(bias.getNumbers(),bias.getChannels(),bias.getHeight(),bias.getWidth());

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
		//计算kernelGradient,这里并不更新kernel,kernel在优化器中更新
		kernelGradient.fillValue(0);
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int c=0;c<inputDiff.getChannels();c++){
				int inputChannelIndex = c/(inputDiff.getChannels()/input.getChannels());
				for(int h=0;h<inputDiff.getHeight();h++){
					for(int w=0;w<inputDiff.getWidth();w++){
						//先定位到输出的位置
						//然后遍历kernel,通过kernel定位输入的位置
						//然后将输入乘以diff
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
		//Blob kernelRoate180 = MathFunctions.rotate180Blob(kernel);
		//然后再做卷积
		outputDiff.fillValue(0);
		MathFunctions.convolutionBlobSame(inputDiff, kernel, outputDiff);	
		
		paramsList.clear();
		paramsList.add(kernel);
		paramsList.add(bias);
		
		gradientList.clear();
		gradientList.add(kernelGradient);
		gradientList.add(biasGradient);
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
			out.writeObject(kernel);
			out.writeObject(bias);
			if(activationFunc != null){
				out.writeUTF(activationFunc.getType());
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	@Override
	public void loadModel(ObjectInputStream in) {
		// TODO Auto-generated method stub
		try {
			kernel = (Blob) in.readObject();
			bias = (Blob) in.readObject();
			String activationType = in.readUTF();
			if(activationType.equals(ReluActivationFunc.TYPE)){
				setActivationFunc(new ReluActivationFunc());
			}else if(activationType.equals(SigmodActivationFunc.TYPE)){
				setActivationFunc(new SigmodActivationFunc());
			}else if(activationType.equals(TanhActivationFunc.TYPE)){
				setActivationFunc(new TanhActivationFunc());
			}
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
