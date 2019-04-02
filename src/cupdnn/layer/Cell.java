package cupdnn.layer;

import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import cupdnn.Network;
import cupdnn.data.Blob;

public class Cell extends Layer{

	public Cell(Network network) {
		super(network);
		// TODO Auto-generated constructor stub
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
	
	public void forward(Blob in,Blob out) {
		
	}
	
	public void backward(Blob in,Blob out,Blob inDiff,Blob outDiff) {
		
	}
	
	public void resetState() {
		
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
