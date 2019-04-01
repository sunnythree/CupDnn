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
import cupcnn.util.ThreadPoolManager;

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
 * y(t) = tanh(c + V*h(t))            (2)
*/
public class RnnCell extends Cell {
	private int inSize;
	private int outSize;
	private int batch;
	private Network mNetwork;
	private Blob Ht_1;
	private Blob Ht_1his;
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
	private Blob z1;
	private ActivationFunc tanh;
	private final String TYPE = "RNN";
	private RecurrentLayer mRecurrentLayer;

	public RnnCell(Network network, RecurrentLayer recurrentLayer) {
		super(network);
		// TODO Auto-generated constructor stub
		this.mNetwork = network;
		this.mRecurrentLayer = recurrentLayer;
		this.tanh = new TanhActivationFunc();
		this.batch = network.getBatch() / mRecurrentLayer.getSeqLen();
	}

	public RnnCell(Network network, RecurrentLayer recurrentLayer, int inSize, int outSize) {
		this(network, recurrentLayer);
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
		return TYPE;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		// 表明该层公有outSize个神经元，每个神经元和前面层的inSize个神经元向连
		if(U==null) {
			U = new Blob(inSize, outSize);
			// 高斯分布初始化
			MathFunctions.gaussianInitData(U.getData());
		}
		
		UW = new Blob(inSize, outSize);

		// 表明该层有outSize个神经元，每个神经元有一个偏执
		if(bias==null) {
			bias = new Blob(outSize);
			// 常量初始化b
			MathFunctions.constantInitData(bias.getData(), 0.0f);
		}
		biasW = new Blob(outSize);

		
		if(W==null) {
			W = new Blob(outSize, outSize);
			// 高斯分布初始化
			MathFunctions.gaussianInitData(W.getData());
		}
		WW = new Blob(outSize, outSize);

		
		if(V==null) {
			V = new Blob(outSize, outSize);
			// 高斯分布初始化
			MathFunctions.gaussianInitData(V.getData());
		}
		VW = new Blob(outSize, outSize);
		
		if(c==null) {
			c = new Blob(outSize);
			MathFunctions.constantInitData(c.getData(), 0.0f);
		}
		cW = new Blob(outSize);

		Ht_1 = new Blob(batch, outSize);
		Ht_1his = new Blob(batch, outSize);
		
		// z是个中间值，计算的时候要用到。
		z = new Blob(batch, outSize);
		z1 = new Blob(batch, outSize);
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub

	}

	public void forward(Blob in, Blob out) {
		float[] inData = in.getData();
		float[] outData = out.getData();
		float[] UData = U.getData();
		float[] WData = W.getData();
		float[] VData = V.getData();
		float[] biasData = bias.getData();
		float[] cData = c.getData();
		float[] Ht_1Data = Ht_1.getData();
		float[] Ht_1hisData = Ht_1his.getData();
		float[] zData = z.getData();
		float[] z1Data = z1.getData();
		
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < outSize; j++) {
				float nextState = 0;
				// U*in
				for (int k = 0; k < inSize; k++) {
					nextState += inData[i * inSize + k] * UData[j * inSize + k];
				}
				// W*ht-1
				for (int k = 0; k < outSize; k++) {
					nextState += Ht_1Data[i * outSize + k] * WData[j * outSize + k];
				}
				// add bias
				nextState += biasData[j];
				zData[i * outSize + j] = nextState;
				// 保存历史的历史，反向传播会用到
				Ht_1hisData[i * outSize + j] = Ht_1Data[i * outSize + j];
				Ht_1Data[i * outSize + j] = tanh.active(nextState);
			}
		}
		// V*ht+b
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < outSize; j++) {
				// W*ht
				float outTmp = 0;
				for (int k = 0; k < outSize; k++) {
					outTmp += VData[j * outSize + k] * Ht_1Data[i * outSize + k];
				}
				// add bias
				outTmp += cData[j];
				z1Data[i * outSize + j] = outTmp;
				outData[i * outSize + j] = tanh.active(outTmp);
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

	public void backward(Blob in, Blob out, Blob inDiff, Blob outDiff) {
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
		float[] Ht_1hisData = Ht_1his.getData();
		float[] zData = z.getData();
		float[] z1Data = z.getData();

		// 先乘激活函数的偏导数,即可求出当前层的误差
		for (int n = 0; n < batch; n++) {
			for (int ids = 0; ids < outSize; ids++) {
				inDiffData[n * outSize + ids] *= tanh.diffActive(z1Data[n * outSize + ids]);
			}
		}

		VW.fillValue(0);
		cW.fillValue(0);
		for (int n = 0; n < batch; n++) {
			for (int ids = 0; ids < outSize; ids++) {
				cWData[ids] += inDiffData[n * outSize + ids];
				for (int is = 0; is < outSize; is++) {
					// 相当于一个神经元和它的每一个连接乘加
					VWData[ids * outSize + is] += Ht_1Data[n * outSize + is] * inDiffData[n * outSize + ids];
				}
			}
		}
		// calculate cWData
		MathFunctions.dataDivConstant(VWData, batch);
		MathFunctions.dataDivConstant(cWData, batch);
		mNetwork.updateW(c, cW);
		mNetwork.updateW(V, VW);
		// 残差继续传播
		Blob tmpDiff = new Blob(batch, outSize);
		tmpDiff.fillValue(0);
		float[] tmpDiffData = tmpDiff.getData();
		for (int n = 0; n < batch; n++) {
			for (int ids = 0; ids < outSize; ids++) {
				for (int ods = 0; ods < outSize; ods++) {
					tmpDiffData[n * outSize + ods] += inDiffData[n * outSize + ids] * VData[ids * outSize + ods];
				}
			}
		}
		// 乘以激活函数导数得到隐藏层残差
		for (int n = 0; n < batch; n++) {
			for (int ids = 0; ids < outSize; ids++) {
				tmpDiffData[n * outSize + ids] *= tanh.diffActive(zData[n * outSize + ids]);
			}
		}
		UW.fillValue(0);
		WW.fillValue(0);
		biasW.fillValue(0);
		// 同时更新隐藏层和输入层的参数
		for (int i = 0; i < batch; i++) {
			for (int j = 0; j < outSize; j++) {
				biasWData[j] += tmpDiffData[i * outSize + j];
				// input*inDiff
				for (int k = 0; k < inSize; k++) {
					UWData[j * inSize + k] += inData[i * inSize + k] * tmpDiffData[i * outSize + j];
				}
				for (int k = 0; k < outSize; k++) {
					WWData[j * outSize + k] += Ht_1hisData[i * outSize + k] * tmpDiffData[i * outSize + j];
				}
			}
		}
		// 平均
		MathFunctions.dataDivConstant(UWData, batch);
		MathFunctions.dataDivConstant(WWData, batch);
		MathFunctions.dataDivConstant(biasWData, batch);

		// 更新参数
		mNetwork.updateW(U, UW);
		mNetwork.updateW(W, WW);
		mNetwork.updateW(bias, biasW);
		// 残差继续传播
		outDiff.fillValue(0);
		for (int n = 0; n < batch; n++) {
			for (int ids = 0; ids < outSize; ids++) {
				for (int ods = 0; ods < inSize; ods++) {
					outDiffData[n * inSize + ods] += tmpDiffData[n * outSize + ids] * UData[ids * inSize + ods];
				}
			}
		}
		// System.out.println(outDiffData[0]);
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
