package cupcnn.layer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupcnn.Network;
import cupcnn.active.ActivationFunc;
import cupcnn.active.ReluActivationFunc;
import cupcnn.active.SigmodActivationFunc;
import cupcnn.active.TanhActivationFunc;
import cupcnn.data.Blob;
import cupcnn.util.MathFunctions;
import cupcnn.util.Task;

/* Computes the following operations:
 * y(t-1)    y(t)
 *   ^        ^
 *   |V+c     | V+c
 * h(t-1) -> h(t)
 *   ^ +b W   ^ +b
 *   |U       |U
 * x(t-1)    x(t)
 *
 * h(t) = tanh(b + W*h(t-1) + U*x(t)) (1)
 * y(t) = c + V*h(t)                  (2)
*/ 
public class RnnCell extends Cell{
	private int inSize;
	private int outSize;
	private int batch;
	private Network mNetwork;
	private Blob Ht_1;
	private Blob U;
	private Blob UW;
	private Blob W;
	private Blob WW;
	private Blob V; 
	private Blob VW;
	private Blob bias;
	private Blob biasW;
	private Blob c;
	private Blob cW;
	private Blob z;
	private ActivationFunc tanh;
	private final String TYPE = "RNN";
	private RecurrentLayer mRecurrentLayer;

	public RnnCell(Network network,RecurrentLayer recurrentLayer) {
		super(network);
		// TODO Auto-generated constructor stub
		this.mNetwork = network;
		this.mRecurrentLayer = recurrentLayer;
		this.tanh = new TanhActivationFunc();
		this.batch = network.getBatch()/recurrentLayer.getSeqLen();
	}
	
	public RnnCell(Network network,RecurrentLayer recurrentLayer,int inSize,int outSize) {
		this(network,recurrentLayer);
		// TODO Auto-generated constructor stub
		this.inSize = inSize;
		this.outSize = outSize;
	}
	

	@Override
	public Blob createOutBlob() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Blob createDiffBlob() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		//表明该层公有outSize个神经元，每个神经元和前面层的inSize个神经元向连
		U = new Blob(inSize,outSize);
		UW = new Blob(inSize,outSize);

		//表明该层有outSize个神经元，每个神经元有一个偏执
		bias = new Blob(outSize);
		biasW = new Blob(outSize);
		
		W = new Blob(outSize,outSize);
		WW = new Blob(outSize,outSize);
		
		V = new Blob(outSize,outSize);
		VW = new Blob(outSize,outSize);
		c = new Blob(outSize);
		cW = new Blob(outSize);
		
		Ht_1 = new Blob(batch,outSize);

		//初始化
		//高斯分布初始化
		MathFunctions.gaussianInitData(U.getData());
		MathFunctions.gaussianInitData(W.getData());
		MathFunctions.gaussianInitData(V.getData());
		//常量初始化b
		MathFunctions.constantInitData(bias.getData(), 0.0f);
		MathFunctions.constantInitData(c.getData(), 0.0f);
		//z是个中间值，计算的时候要用到。
		z = new Blob(batch,outSize);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		
	}

	public void forward(Blob in,Blob out) {
		float[] inData = in.getData();
		float[] outData = out.getData();
		float[] UData = U.getData();
		float[] WData = W.getData();
		float[] VData = V.getData();
		float[] biasData = bias.getData();
		float[] cData = c.getData();
		float[] Ht_1Data = Ht_1.getData();
		float[] zData = z.getData();
		for(int i=0;i<batch;i++) {
			for(int j=0;j<outSize;j++) {
				float nextState = 0;
				//U*in
				for(int k=0;k<inSize;k++) {
					nextState += inData[i*inSize+k]*UData[j*inSize+k];
				}
				//W*ht-1
				for(int k=0;k<outSize;k++) {
					nextState += Ht_1Data[i*outSize+k]*WData[j*outSize+k];
				}
				//add bias
				nextState += biasData[j];
				zData[i*outSize+j] = nextState;
				Ht_1Data[i*outSize+j] = tanh.active(nextState);
			}
		}
		//V*ht+b
		for(int i=0;i<batch;i++) {
			float outTmp = 0;
			for(int j=0;j<outSize;j++) {
				//W*ht
				for(int k=0;k<outSize;k++) {
					outTmp += VData[j*outSize+k]*Ht_1Data[i*outSize+k];
				}
				//add bias
				outTmp += cData[j];
			}
		}
	}
	
	@Override
	public void resetState() {
		Ht_1.fillValue(0);
	}
	
	@Override
	public void backward() {
		// TODO Auto-generated method stub
		
	}
	
	public void backward(Blob in,Blob out,Blob inDiff,Blob outDiff) {
		float[] inData = in.getData();
		float[] outData = out.getData();
		float[] inDiffData = inDiff.getData();
		float[] outDiffData = outDiff.getData();
		float[] UData = U.getData();
		float[] WData = W.getData();
		float[] VData = V.getData();
		float[] UWData = UW.getData();
		float[] WWData = WW.getData();
		float[] VWData = VW.getData();
		float[] biasWData = biasW.getData();
		float[] cWData = cW.getData();
		float[] Ht_1Data = Ht_1.getData();
		float[] zData = z.getData();
		
		
		for(int n = 0; n < batch; n++){
			for(int ids = 0; ids < outSize; ids++){
				for(int is = 0; is < inSize; is++){
					//相当于一个神经元和它的每一个连接乘加
					VWData[ids*inSize+is] += Ht_1Data[n*inSize+is] * inDiffData[n*outSize+ids];
				}
			}
		}
		MathFunctions.dataDivConstant(VWData, batch);
		MathFunctions.dataDivConstant(cWData, batch);
		mNetwork.updateW(c, cW);
		mNetwork.updateW(V, VW);
		//残差继续传播
		outDiff.fillValue(0);
		for(int n = 0; n < batch;n++){
			for(int ids = 0; ids < outSize; ids++){
				for(int ods = 0; ods < inSize; ods++){
					outDiffData[n*inSize+ods] += inDiffData[n*outSize+ids]*VData[ids*inSize+ods];
				}
			}
		}
		System.arraycopy(outDiffData, 0, inDiffData, 0, outDiffData.length);
		
		for(int n=0; n < batch;n++){
			for(int ids = 0; ids < outSize; ids++){
				inDiffData[n*outSize+ids] *= tanh.diffActive(zData[n*outSize+ids]);
			}
		}
		
		for(int i=0;i<batch;i++) {
			for(int j=0;j<outSize;j++) {
				biasWData[i*outSize+j] = inDiffData[i*outSize+j];
				//input*inDiff
				for(int k=0;k<inSize;k++) {
					UWData[j*inSize+k] += inData[i*inSize+k]*inDiffData[i*outSize+j];
				}
				for(int k=0;k<inSize;k++) {
					WWData[j*inSize+k] += inData[i*inSize+k]*inDiffData[i*outSize+j];
				}
			}
		}
		//平均
		MathFunctions.dataDivConstant(UWData, batch);
		MathFunctions.dataDivConstant(WWData, batch);
		MathFunctions.dataDivConstant(biasWData, batch);
		
		//更新参数
		mNetwork.updateW(U, UW);
		mNetwork.updateW(W, WW);
		mNetwork.updateW(bias, biasW);	
		//残差继续传播
		for(int n = 0; n < batch;n++){
			for(int ids = 0; ids < outSize; ids++){
				for(int ods = 0; ods < inSize; ods++){
					outDiffData[n*inSize+ods] += inDiffData[n*outSize+ids]*(UData[ids*inSize+ods]+WWData[ids*inSize+ods]);
				}
			}
		}
	}

	@Override
	public void saveModel(ObjectOutputStream out) {
		// TODO Auto-generated method stub
		try {
			out.writeUTF(getType());
			out.writeInt(inSize);
			out.writeInt(outSize);
			out.writeObject(U);
			out.writeObject(W);
			out.writeObject(V);
			out.writeObject(bias);
			out.writeObject(c);
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
			U = (Blob) in.readObject();
			W = (Blob) in.readObject();
			V = (Blob) in.readObject();
			bias = (Blob) in.readObject();
			c = (Blob) in.readObject();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
