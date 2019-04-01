package cupcnn.layer;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Vector;

import cupcnn.Network;
import cupcnn.data.Blob;
import cupcnn.data.BlobParams;
import cupcnn.util.Task;
import cupcnn.util.ThreadPoolManager;

public class SoftMaxLayer extends Layer{
	public static final String TYPE = "SoftMaxLayer";
	
	private int size;
	
	public SoftMaxLayer(Network network) {
		super(network);
		// TODO Auto-generated constructor stub
	}

	public SoftMaxLayer(Network network,int size) {
		super(network);
		// TODO Auto-generated constructor stub
		this.size = size;
	}

	
	@Override
	public String getType() {
		// TODO Auto-generated method stub
		return TYPE;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void forward() {
		// TODO Auto-generated method stub
		Blob input = mNetwork.getDatas().get(id-1);
		Blob output = mNetwork.getDatas().get(id);
		float[] inputData = input.getData();
		float[] outputData = output.getData();
		assert input.getSize()==output.getSize():"SoftMax forward---- input.getSize()==output.getSize() error";
	
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		int batch = mNetwork.getBatch();
		for(int n=0;n<batch;n++){
			workers.add(new Task<Object>(n) {
				@Override
			    public Object call() throws Exception {
					float sum = 0.0f;
					float max = 0.001f;
					
					//查找最大值
					for(int is=0;is<input.getWidth();is++){
						max = Math.max(max, inputData[n*input.getWidth()+is]);
					}
					//求和
					for(int is=0;is<input.getWidth();is++){
						outputData[n*input.getWidth()+is] = (float) Math.exp(inputData[n*input.getWidth()+is]-max);
						sum += outputData[n*input.getWidth()+is];
					}
					if(sum==0){
						System.out.println("sum is zero");
						System.exit(0);
					}
					//每一项除以sum
					for(int os=0;os<output.getWidth();os++){
						outputData[n*output.getWidth()+os] /= sum;
					}
					
//					//求和
//					for(int is=0;is<input.get3DSize();is++){
//						sum += Math.exp(inputData[n*input.get3DSize()+is]);
//					}
//					//每一项除以sum
//					for(int os=0;os<output.get3DSize();os++){
//						outputData[n*output.get3DSize()+os] = Math.exp(inputData[n*output.get3DSize()+os])/sum;
//					}
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
		Blob output = mNetwork.getDatas().get(id);
		float[] inputDiffData = inputDiff.getData();
		float[] outputDiffData = outputDiff.getData();
		float[] outputData = output.getData();
		assert inputDiff.getSize()==outputDiff.getSize():"SoftMax backward---- inputDiff.getSize()==outputDiff.getSize() error";
		
		//先求softmax函数的偏导数
		outputDiff.fillValue(0);
		Vector<Task<Object>> workers = new Vector<Task<Object>>();
		int batch = mNetwork.getBatch();
		for(int n=0;n<batch;n++){
			workers.add(new Task<Object>(n) {
				@Override
			    public Object call() throws Exception {
					for(int ods=0;ods<outputDiff.getWidth();ods++){
						for(int ids=0;ids<inputDiff.getWidth();ids++){
							if(ids==ods){
								outputDiffData[n*output.getWidth()+ods] += outputData[n*output.getWidth()+ods]*(1.0-outputData[n*output.getWidth()+ods])
										*inputDiffData[n*output.getWidth()+ids];
							}else{
								outputDiffData[n*output.getWidth()+ods] -= outputData[n*output.getWidth()+ods]*outputData[n*output.getWidth()+ids]
										*inputDiffData[n*output.getWidth()+ids];
							}
						}
					}
					return null;
				}
			});
		}
		ThreadPoolManager.getInstance(mNetwork).dispatchTask(workers);
	}

	@Override
	public void saveModel(ObjectOutputStream out) {
		// TODO Auto-generated method stub
		try {
			out.writeUTF(getType());
			out.writeInt(size);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public void loadModel(ObjectInputStream in) {
		// TODO Auto-generated method stub
		try {
			size = in.readInt();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	@Override
	public Blob createOutBlob() {
		// TODO Auto-generated method stub
		return new Blob(mNetwork.getBatch(),size);
	}

	@Override
	public Blob createDiffBlob() {
		// TODO Auto-generated method stub
		return new Blob(mNetwork.getBatch(),size);
	}

}
