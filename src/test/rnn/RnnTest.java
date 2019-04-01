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
			float[] data1 = new float[1];
			data1[0] = a;
			float[] data2 = new float[1];
			data2[0] = b;
			float[] label1 = new float[1];
			label1[0] = a;
			float[] label2 = new float[1];
			label2[0] = a+b;
			DataAndLabel tmp1 = new DataAndLabel(1,1);
			tmp1.setData(data1, label1);
			dals.add(tmp1);
			DataAndLabel tmp2 = new DataAndLabel(1,1);
			tmp2.setData(data2, label2);
			dals.add(tmp2);
		}
		return dals;
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		AddNetwork aw = new AddNetwork();	
		aw.buildNetwork();
		aw.train(genDatas(50000), 30);
	
        Scanner sc = new Scanner(System.in);   
        
        System.out.println("please input two numbers(-1~1)");  
		System.out.println("please input first one:");
        float a = Float.parseFloat(sc.next());
        System.out.println("please input second one:");
        float b = Float.parseFloat(sc.next());
        
        Blob in = new Blob(1,1,1,1);
        float[] inData = in.getData();
        inData[0] = a;
        aw.predict(in);
        inData[0] = b;
        Blob result = aw.predict(in);
        float[] resultData = result.getData();
        System.out.println(a+" + "+b+" = "+resultData[0]);
	}
}
