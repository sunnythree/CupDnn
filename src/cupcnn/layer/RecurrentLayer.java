package cupcnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupcnn.Network;
import cupcnn.data.Blob;

public class RecurrentLayer extends Layer {
	Network mNetwork;
	Layer mCell;
	int seqLen;
	int batch;
	int inSize;
	int outSize;
	
	enum RecurrentType{
		RNN,
		LSTM,
		GRU
	}
	
	RecurrentType type;

	public RecurrentLayer(Network network) {
		super(network);
		// TODO Auto-generated constructor stub
	}
	
	public RecurrentLayer(Network network,RecurrentType type,int seqLen,int inSize,int outSize) {
		super(network);
		this.mNetwork = network;
		this.batch = network.getBatch();
		this.inSize = inSize;
		this.outSize = outSize;
		this.type = type;
	}

	@Override
	public Blob createOutBlob() {
		// TODO Auto-generated method stub
		return new Blob(seqLen,batch,outSize);
	}

	@Override
	public Blob createDiffBlob() {
		// TODO Auto-generated method stub
		return new Blob(seqLen,batch,outSize);
	}

	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return "RecurrentLayer";
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		switch(type) {
		case RNN:
			mCell = new RnnCell(mNetwork);
			break;
		case LSTM:
			break;
		case GRU:
			break;
		}
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		for(int i=0;i<seqLen;i++) {
			
		}
	}

	@Override
	public void backward() {
		// TODO Auto-generated method stub
		
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
