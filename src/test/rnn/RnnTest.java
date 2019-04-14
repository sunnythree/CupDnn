package test.rnn;

import java.util.Random;
import java.util.Scanner;
import java.util.Vector;

import cupcnn.Network;
import cupcnn.data.Blob;
import cupcnn.optimizer.SGDOptimizer;
import cupcnn.util.DataAndLabel;

public class RnnTest {
	
	public static Vector<DataAndLabel>genDatas(int samples) {
		Vector<DataAndLabel> dals = new Vector<DataAndLabel>();
		Random random = new Random();
		for(int i=0;i<samples;i++) {
			float a = random.nextFloat();
			//将数据扩充到-1-->1
			if(random.nextBoolean()) {
				a = -a;
			}
			float b = random.nextFloat();
			if(random.nextBoolean()) {
				b = -b;
			}
			float[] data = new float[2];
			data[0] = a;
			data[1] = b;
			float[] label = new float[2];
			label[0] = a;
			label[1] = a+b;
			DataAndLabel tmp = new DataAndLabel(2,2);
			tmp.setData(data, label);
			dals.add(tmp);
		}
		return dals;
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		AddNetwork aw = new AddNetwork();	
		aw.buildNetwork();
		aw.train(genDatas(10000), 20);
		aw.saveModel("model/rnn_add.model");
	
		aw.loadModel("model/rnn_add.model");
        Scanner sc = new Scanner(System.in);   
        while(true) {
		    System.out.println("please input two numbers(-1~1),input q to quit");  
			System.out.println("please input first one:");
			String tmp = sc.next();
			if(tmp.equals("q")) {
				break;
			}
		    float a = Float.parseFloat(tmp);
		    System.out.println("please input second one:");
		    tmp = sc.next();
			if(tmp.equals("q")) {
				break;
			}
		    float b = Float.parseFloat(tmp);
		    
		    Blob in = new Blob(1,1,1,2);
		    float[] inData = in.getData();
		    inData[0] = a;
		    inData[1] = b;
		    Blob result = aw.predict(in);
		    float[] resultData = result.getData();
		    System.out.println(a+" + "+b+" = "+resultData[1]);
        }
        System.out.println("rnn done");
	}
}
