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
			float[] data = new float[2];
			data[0] = random.nextFloat();
			//将数据扩充到-1-->1
			if(random.nextBoolean()) {
				data[0] = -data[0];
			}
			data[1] = random.nextFloat();
			if(random.nextBoolean()) {
				data[1] = -data[1];
			}
			float[] label = new float[1];
			label[0] = data[0]+data[1];
			DataAndLabel tmp = new DataAndLabel(2,1);
			tmp.setData(data, label);
			dals.add(tmp);
		}
		return dals;
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		AddNetwork aw = new AddNetwork();	
		aw.buildNetwork();
		aw.train(genDatas(10000), 30);
	
        Scanner sc = new Scanner(System.in);   
        
        System.out.println("please input two numbers(-1~1)");  
		System.out.println("please input first one:");
        float a = Float.parseFloat(sc.next());
        System.out.println("please input second one:");
        float b = Float.parseFloat(sc.next());
        
        Blob in = new Blob(1,1,1,2);
        float[] inData = in.getData();
        inData[0] = a;
        inData[1] = b;
        Blob result = aw.predict(in);
        float[] resultData = result.getData();
        System.out.println("result is : "+resultData[0]);
	}
}
