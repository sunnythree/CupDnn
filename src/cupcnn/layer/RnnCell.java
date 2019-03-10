package cupcnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupcnn.Network;
import cupcnn.data.Blob;

public class RnnCell extends Layer{
	private int inSize;
	private int outSize;
	private int batch;
	private Network mNetwork;
	private Layer mRecurrentLayer;

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
		this.mRecurrentLayer = recurrentLayer;
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
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
	
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
