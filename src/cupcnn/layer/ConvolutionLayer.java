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
		//layerParams.getHeight()��ʾ�ò���Ҫ��ȡ����������
		if(kernel ==null && bias == null){
			kernel = new Blob(kernelParams.getNumbers(),kernelParams.getChannels(),kernelParams.getHeight(),kernelParams.getWidth());
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
		//�����Ľ��������z��
		z.fillValue(0);
		MathFunctions.convolutionBlobSame(input, kernel, bias, z);
		//�����
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
		
		//�ȳ˼�����ĵ���,�õ��ò�����
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
		
		//Ȼ����²���
		//����kernelGradient,���ﲢ������kernel,kernel���Ż����и���
		kernelGradient.fillValue(0);
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int c=0;c<inputDiff.getChannels();c++){
				int inputChannelIndex = c/(inputDiff.getChannels()/input.getChannels());
				for(int h=0;h<inputDiff.getHeight();h++){
					for(int w=0;w<inputDiff.getWidth();w++){
						//�ȶ�λ�������λ��
						//Ȼ�����kernel,ͨ��kernel��λ�����λ��
						//Ȼ���������diff
						int inStartX = w - kernelGradient.getWidth()/2;
						int inStartY = h - kernelGradient.getHeight()/2;
						//�;���˳˼�
			
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
		//ƽ��
		MathFunctions.dataDivConstant(kernelGradientData, inputDiff.getNumbers());
		
		//����bias
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
		//ƽ��
		MathFunctions.dataDivConstant(biasGradientData, inputDiff.getNumbers());
		
		if(id<=1)return;
		//�Ȱ�kernel��ת180��
		//Blob kernelRoate180 = MathFunctions.rotate180Blob(kernel);
		//Ȼ���������
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
			//�����ʱ��batchҲ����layerParams��number����1����Ϊpredict��ʱ����Ϊ����ʹ�õ�ʱ�����batchһ�㶼��1
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
