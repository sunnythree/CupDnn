package test;


import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.List;

import cupcnn.Network;
import cupcnn.active.ReluActivationFunc;
import cupcnn.layer.FullConnectionLayer;
import cupcnn.loss.CrossEntropyLoss;
import cupcnn.optimizer.SGDOptimizer;
import cupcnn.data.Blob;

public class MnistTest {
	static List<DigitImage> trains = null ;
	static List<DigitImage> tests = null;
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		
		//load mnist
		ReadFile rf1=new ReadFile("data\\train-labels.idx1-ubyte","data\\train-images.idx3-ubyte");
		ReadFile rf2=new ReadFile("data\\t10k-labels.idx1-ubyte","data\\t10k-images.idx3-ubyte");
		try {
			tests = rf2.loadDigitImages();
			trains =rf1.loadDigitImages();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		MnistNetwork mn = new MnistNetwork();
		mn.buildNetwork();
		mn.train(trains,30);
		mn.test(tests);
	}

}
