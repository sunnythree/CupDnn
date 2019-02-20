package test.cifar10;


import java.io.IOException;
import java.util.List;

import cupcnn.util.DigitImage;
import test.cifar10.Cifar10Network;
import test.cifar10.ReadFile;

	public class Cifar10Test {
	static List<DigitImage> trains = null ;
	static List<DigitImage> tests = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		//load cifar10
		ReadFile rf1=new ReadFile("data/cifar10/data_batch_%.bin");
		ReadFile rf2=new ReadFile("data/cifar10/test_batch.bin");
		try {
			tests = rf2.loadDigitImages(false);
			trains =rf1.loadDigitImages(true);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Cifar10Network mn = new Cifar10Network();
		mn.buildNetwork(trains.size());
		mn.train(trains,10,tests);
		//mn.test(tests);
		mn.saveModel("model/cifar10.model");
		
		
		mn.loadModel("model/cifar10.model");
		mn.test(tests);

	}

}
