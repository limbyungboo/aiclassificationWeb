package kr.co.aiweb.machinelearning.trainmodel;

import java.io.File;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import kr.co.aiweb.machinelearning.common.MLConst.MLDatasetConst;
import kr.co.aiweb.machinelearning.listener.RealTimeTrainingListener;
import kr.co.aiweb.machinelearning.vo.LabelInfo;
import kr.co.aiweb.machinelearning.vo.PredictResult;
import lombok.extern.slf4j.Slf4j;

/**
 * deeplearning Vgg16 model trainer
 */
@Slf4j
public class TrainModel_ResNet50 extends TrainModel{

	/**
	 * model
	 */
	private ComputationGraph model;
	
	/**
	 * FineTuneConfiguration
	 */
	private FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
												.updater(new Adam(1e-4))
												.seed(MLDatasetConst.SEED.getValue())
												.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
												.build();
	
	/**constructor
	 * @param rootDir
	 * @throws Exception
	 */
	protected TrainModel_ResNet50(File rootDir) throws Exception {
		super(rootDir);
	}
	
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#initModel()
	 */
	@Override
	protected void initModel() throws Exception {
		super.modelFile = new File(rootDir, "resnet50_model.zip");
		super.labelFile = new File(rootDir, "resnet50_label.json");
		super.statFile = new File(rootDir, "resnet50_stat.dl4j");
		preProcessor = new VGG16ImagePreProcessor();
		
		//label 정보 로드
		labelInfo = new LabelInfo(labelFile);
		
		//모델파일이 존재
		if(modelFile.exists() == true) {
			log.info(".............. 저장 학습모델 loading....... ");
			model = ModelSerializer.restoreComputationGraph(modelFile);
		}
	}
    
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#updateModel(int)
	 */
	@SuppressWarnings("unchecked")
	@Override
	protected void updateModel(int nOut) throws Exception {
		ComputationGraph baseModel = null;
		if(model == null) {
			ZooModel<ResNet50> zooModel = ResNet50.builder().build();
			baseModel = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
			log.info("----- load successed original ResNet50 model");
		}
		else {
			Layer outputLayer = model.getLayer("fc1000");
			if (outputLayer.conf().getLayer() instanceof OutputLayer) {
			    OutputLayer ol = (OutputLayer) outputLayer.conf().getLayer();
				//기존 out 가 동일할경우
			    if(nOut == ol.getNOut()) {
					//set listeners
					setListeners();
					return;
				}
			}
			baseModel = model;
		}
		
		log.info("=================>>> ComputationGraph model update");
		//ResNet50 마지막 layer 는 fc1000 이것을 정의한 Outputlayer 로 교체..
		//fc1000 이전층(flatten_1)은 유지
		model = new TransferLearning.GraphBuilder(baseModel)
				.fineTuneConfiguration(fineTuneConf) // 여기 추가
	            .setFeatureExtractor("flatten_1")  // 마지막 flatten_1 전까지 freeze
	            .removeVertexAndConnections("fc1000")
                .addLayer("fc1000",new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(2048)
                        .nOut(nOut)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build(),"flatten_1") //add in a final output dense layer,
	            .setOutputs("fc1000")
	            .build();
		
		//set listeners
		setListeners();
	}
    
	/**
	 * set listeners
	 */
	private void setListeners() {
		if(model.getListeners().size() == 0) {
			StatsStorage statsStorage = new FileStatsStorage(super.statFile);
			model.setListeners(
				  new ScoreIterationListener(MLDatasetConst.EPOCHS.getValue())
				, new RealTimeTrainingListener()
				, new StatsListener(statsStorage)
			);
		}
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
		model.fit(datasetIter, MLDatasetConst.EPOCHS.getValue());
		
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
