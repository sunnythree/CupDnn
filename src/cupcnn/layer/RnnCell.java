package cupcnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupcnn.Network;
import cupcnn.active.ActivationFunc;
import cupcnn.active.TanhActivationFunc;
import cupcnn.data.Blob;
import cupcnn.util.MathFunctions;

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

	public RnnCell(Network network) {
		super(network);
		// TODO Auto-generated constructor stub
	}
	
	public RnnCell(Network network,Layer recurrentLayer,int inSize,int outSize) {
		super(network);
		// TODO Auto-generated constructor stub
		this.inSize = inSize;
		this.outSize = outSize;
		this.batch = network.getBatch();
		this.mNetwork = network;
		this.tanh = new TanhActivationFunc();
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

		//初始化
		//高斯分布初始化
		MathFunctions.gaussianInitData(U.getData());
		MathFunctions.gaussianInitData(W.getData());
		MathFunctions.gaussianInitData(V.getData());
		//常量初始化b
		MathFunctions.constantInitData(bias.getData(), 0.0f);
		MathFunctions.constantInitData(c.getData(), 0.0f);
		//z是个中间值，计算的时候要用到。
		z = new Blob(mNetwork.getBatch(),outSize);
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
				outData[i*outSize+j] = outTmp;
			}
		}
	}
	
	@Override
	public void backward() {
		// TODO Auto-generated method stub
		
	}
	
	public void backward(Blob in,Blob out,Blob inDiff,Blob outDiff) {
		
	}

	@Override
	public void saveModel(ObjectOutputStream out) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void loadModel(ObjectInputStream in) {
		// TODO Auto-generated method stub
		
	}

}
