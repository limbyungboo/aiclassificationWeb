/**
 * 
 */
package kr.co.aiweb.machinelearning.trainmodel;

import java.io.File;

import org.apache.commons.lang3.StringUtils;

/**
 * 
 */
public class TrainModelFactory {

	/**
	 * Vgg16 model
	 */
	private static TrainModel _vgg16;
	
	/**
	 * MultiLayerNetwork model
	 */
	private static TrainModel _multilayer;
	
	/**
	 * MobileNetV2
	 */
	private static TrainModel _resnet50;
	
	private static String _model;
	
	/**초기화
	 * @param rootDir
	 */
	public static void initTrainModels(File rootDir, String model) throws Exception {
		if(StringUtils.isNotBlank(_model) == true) {
			return;
		}
		_model = model; 
		if("multilayer".equalsIgnoreCase(model) == true) {
			if(_multilayer == null) {
				_multilayer = new TrainModel_MultiLayer(rootDir);
			}
		}
		else if("resnet50".equalsIgnoreCase(model) == true) {
			if(_resnet50 == null) {
				_resnet50 = new TrainModel_ResNet50(rootDir);
			}
		}
		else {
			if(_vgg16 == null) {
				_vgg16 = new TrainModel_Vgg16(rootDir);
			}
		}
	}
	
	/**get model
	 * @return
	 */
	public static TrainModel model() {
		if("multilayer".equalsIgnoreCase(_model) == true) {
			return _multilayer;
		}
		else if("resnet50".equalsIgnoreCase(_model) == true) {
			return _resnet50;
		}
		else {
			return _vgg16;
		}
	}
	
}
