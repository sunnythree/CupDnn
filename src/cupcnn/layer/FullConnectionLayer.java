package cupcnn.layer;

import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.util.MathFunctions;
import cupcnn.util.Task;
import cupcnn.util.ThreadPoolManager;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.Vector;

import cupcnn.Network;
import cupcnn.active.ReluActivationFunc;
import cupcnn.active.SigmodActivationFunc;
import cupcnn.active.TanhActivationFunc;

public class FullConnectionLayer extends Layer{
	public static final String TYPE = "FullConnectionLayer";
	private Blob w;
	private transient Blob wGradient;
	private Blob b;
	private transient Blob bGradient;
	private transient Blob z;
	private int inSize;
	private int outSize;
	
	public FullConnectionLayer(Network network){
		super(network);
	}
	
	public FullConnectionLayer(Network network,int inSize,int outSize){
		super(network);
		this.inSize = inSize;
		this.outSize = outSize;
	}
	
	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		if(w==null && b==null){
			//表明该层公有output.get3DSize()个神经元，每个神经元和前面层的input.get3DSize()个神经元向连
			w = new Blob(outSize,inSize,1,1);

			//表明公有output.getChannels()个神经元，每个神经元有一个偏执
			b = new Blob(outSize,1,1,1);


			//初始化
			double[] wData = w.getData();
			double[] bData = b.getData();
			//高斯分布初始化w
			MathFunctions.gaussianInitData(wData);
			//常量初始化b
			MathFunctions.constantInitData(bData, 0.1);
		}
		assert w!=null && b!=null:"FullConnectionLayer prepare---w or b is null error";
		wGradient = new Blob(w.getNumbers(),w.getChannels(),1,1);
		bGradient = new Blob(b.getNumbers(),b.getChannels(),1,1);
		//z是个中间值，计算的时候要用到。
		z = new Blob(mNetwork.getBatch(),outSize,1,1);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		Blob input = mNetwork.getDatas().get(id-1);
		Blob output = mNetwork.getDatas().get(id);
		double[] inputData = input.getData();
		double[] outputData = output.getData();
		double[] wData = w.getData();
		double[] bData = b.getData();
		double[] zData = z.getData();
		z.fillValue(0);
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		for(int n=0;n<input.getNumbers();n++){
			workers.add(new Task<Object>(n) {
				@Override
			    public Object call() throws Exception {
					for(int os=0;os<output.get3DSize();os++){//有多少个输出，当前层就有多少个神经元
						//和每个神经元的权重相乘
						for(int is=0;is<input.get3DSize();is++){
							//zData[n*output.get3DSize()+os] 表示一个批次中的第n个的第os个神经元
							zData[n*output.get3DSize()+os] += inputData[n*input.get3DSize()+is]*wData[os*input.get3DSize()+is];
						}
						//偏执
						zData[n*output.get3DSize()+os] += bData[os];
						//激活函数
						if(activationFunc!=null){
							outputData[n*output.get3DSize()+os] = activationFunc.active(zData[n*output.get3DSize()+os]);
						}
					}
					return null;
				}
			});
		}
		ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		Blob inputDiff = mNetwork.getDiffs().get(id);
		Blob outputDiff = mNetwork.getDiffs().get(id-1);
		Blob input = mNetwork.getDatas().get(id-1);
		double[] inputData = input.getData();
		double[] inputDiffData = inputDiff.getData();
		double[] outputDiffData = outputDiff.getData();
		double[] wData = w.getData();
		double[] wGradientData = wGradient.getData();
		double[] bGradientData = bGradient.getData();
		double[] zData = z.getData();
		
		//update diff
		//先乘激活函数的偏导数,即可求出当前层的误差
		assert inputDiff.getSize()==z.getSize():"inputDiff.getSize()==z.getSize() error";
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		if(activationFunc != null){
			for(int n=0; n < inputDiff.getNumbers();n++){
				workers.add(new Task<Object>(n) {
					@Override
				    public Object call() throws Exception {
						for(int ids = 0; ids < inputDiff.get3DSize(); ids++){
							inputDiffData[n*inputDiff.get3DSize()+ids] *= activationFunc.diffActive(zData[n*inputDiff.get3DSize()+ids]);
						}
						return null;
					}
				});
			}
			ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
		}

		//update weight
		wGradient.fillValue(0);
		workers.clear();
		for(int n = 0; n < inputDiff.getNumbers(); n++){
			workers.add(new Task<Object>(n) {
				@Override
			    public Object call() throws Exception {
					for(int ids = 0; ids < inputDiff.get3DSize(); ids++){
						for(int is = 0; is < input.get3DSize(); is++){
							//相当于一个神经元和它的每一个连接乘加
							wGradientData[ids*input.get3DSize()+is] += inputData[n*input.get3DSize()+is] * inputDiffData[n*inputDiff.get3DSize()+ids];
						}
					}
					return null;
				}
			});
		}
		ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
		//平均
		MathFunctions.dataDivConstant(wGradientData, input.getNumbers());
		
		//update bias
		bGradient.fillValue(0);
		for(int n=0;n<inputDiff.getNumbers();n++){
			for(int bs = 0; bs < bGradient.getSize(); bs++){
				bGradientData[bs] += inputDiffData[n*bGradient.getSize()+bs];
			}
		}

		//平均
		MathFunctions.dataDivConstant(bGradientData, input.getNumbers());
		
		//最后，乘以当前层的权重后输出
		//每一个输出=每一个神经元与连接他的权重的乘加
		if(id<=1)return;
		outputDiff.fillValue(0);
		workers.clear();
		for(int n = 0; n < outputDiff.getNumbers();n++){
			workers.add(new Task<Object>(n) {
				@Override
			    public Object call() throws Exception {
					for(int ids = 0; ids < inputDiff.get3DSize(); ids++){
						for(int ods = 0; ods < outputDiff.get3DSize(); ods++){
							outputDiffData[n*outputDiff.get3DSize()+ods] += inputDiffData[n*inputDiff.get3DSize()+ids]*wData[ids*w.get3DSize()+ods];
						}
					}
					return null;
				}
			});
		}	
		ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
		
		paramsListW.clear();
		paramsListW.add(w);
		paramsListB.clear();
		paramsListB.add(b);
		
		gradientListW.clear();
		gradientListW.add(wGradient);
		gradientListB.clear();
		gradientListB.add(bGradient);

	}


	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return TYPE;
	}

	@Override
	public void saveModel(ObjectOutputStream out) {
		// TODO Auto-generated method stub
		try {
			out.writeUTF(getType());
			out.writeInt(inSize);
			out.writeInt(outSize);
			out.writeObject(w);
			out.writeObject(b);
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
			inSize = in.readInt();
			outSize = in.readInt();
			w = (Blob) in.readObject();
			b = (Blob) in.readObject();
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

	@Override
	public Blob createOutBlob() {
		// TODO Auto-generated method stub
		return new Blob(mNetwork.getBatch(),outSize,1,1);
	}

	@Override
	public Blob createDiffBlob() {
		// TODO Auto-generated method stub
		return new Blob(mNetwork.getBatch(),outSize,1,1);
	}



}
