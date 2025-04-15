package kr.co.aiweb.machinelearning.trainmodel;

import java.io.File;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import kr.co.aiweb.machinelearning.common.MLConst.DatasetConst;
import kr.co.aiweb.machinelearning.vo.LabelInfo;
import kr.co.aiweb.machinelearning.vo.PredictResult;
import lombok.extern.slf4j.Slf4j;

/**
 * deeplearning Vgg16 model trainer
 */
@Slf4j
public class TrainModel_Vgg16 extends TrainModel{

	/**
	 * model
	 */
	private ComputationGraph model;
	
	/**
	 * FineTuneConfiguration
	 */
	private FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
		    															.updater(new Adam(1e-4))
		    															.seed(DatasetConst.SEED.getValue())
		    															.activation(Activation.RELU)
		    															.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
		    															.build();
    
	/**constructor
	 * @param rootDir
	 * @throws Exception
	 */
	protected TrainModel_Vgg16(File rootDir) throws Exception {
		super(rootDir);
	}
	
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#initModel()
	 */
	@Override
	protected void initModel() throws Exception {
		
		super.modelFile = new File(rootDir, "classification_model_vgg16.zip");
		super.labelFile = new File(rootDir, "classification_label_vgg16.json");
		preProcessor = new VGG16ImagePreProcessor();
		
		//label 정보 로드
		labelInfo = new LabelInfo(labelFile);
		
		//모델파일이 존재
		if(modelFile.exists() == true) {
			model = ModelSerializer.restoreComputationGraph(modelFile);
		}
	}
    
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#updateModel(int)
	 */
	@Override
	protected void updateModel(int nOut) throws Exception {
		ComputationGraph baseModel = null;
		if(model == null) {
			baseModel = (ComputationGraph) VGG16.builder().build().initPretrained(PretrainedType.IMAGENET);
			log.info("----- load successed original vgg16 model");
		}
		else {
			Layer outputLayer = model.getLayer("predictions");
			if (outputLayer.conf().getLayer() instanceof OutputLayer) {
			    OutputLayer ol = (OutputLayer) outputLayer.conf().getLayer();
				//기존 out 가 동일할경우
			    if(nOut == ol.getNOut()) {
					return;
				}
			}
			
			baseModel = model;
		}
		
		log.info("=================>>> ComputationGraph model update");
		model = new TransferLearning.GraphBuilder(baseModel)
        		.fineTuneConfiguration(fineTuneConf) // 여기 추가
                .setFeatureExtractor("fc2") // fc2까지는 freeze
                .removeVertexAndConnections("predictions")
                .addLayer("predictions", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4096)
                        .nOut(nOut)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .build(), "fc2")
                .setOutputs("predictions")
                .build();
	}
    
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#fit(java.io.File)
	 */
	@Override
	public void fit(File datasetDir) throws Exception {
		log.info("---------------------------------- dataset create.");
		DataSetIterator datasetIter = super.createDatasetIterator(datasetDir);
		
		log.info("---------------------------------- model update check :: label count = {}", labelInfo.getLabelCount());
		updateModel(labelInfo.getLabelCount());
		
		log.info("---------------------------------- start training.");
		model.fit(datasetIter, DatasetConst.EPOCHS.getValue());
		
		log.info("---------------------------------- model save.");
		modelSave();
		log.info("---------------------------------- model training is over.");
	}
	
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#predict(org.nd4j.linalg.api.ndarray.INDArray)
	 */
	@Override
	public PredictResult predict(INDArray features) throws Exception {
		if(modelFile.exists() == false) {
			throw new Exception("아직 학습을 진행하지 않았습니다. 먼저 학습을 진행해 주세요");
		}
		
		log.info("................. VGG16ImagePreProcessor transform");
        preProcessor.transform(features);
		
        // 예측
		log.info("................. model outputSingle");
        INDArray output = model.outputSingle(false, features);
        
        //예측 결과 return
		log.info("................. return PredictResult");
        return new PredictResult(labelInfo, output);
	}
	
	/**모델 저장
	 * @throws Exception
	 */
	private void modelSave() throws Exception {
		//모델 저장
		model.save(modelFile);
		
		//label 정보 저장
		labelInfo.save();
	}	
}
