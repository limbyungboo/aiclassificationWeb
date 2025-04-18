package kr.co.aiweb.machinelearning.trainmodel;

import java.io.File;
import java.util.List;
import java.util.stream.Collectors;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

//import ai.djl.modality.Classifications;
//import ai.djl.modality.cv.Image;
//import ai.djl.modality.cv.translator.ImageClassificationTranslator;
//import ai.djl.repository.zoo.Criteria;
import kr.co.aiweb.machinelearning.common.MLConst.DatasetConst;
import kr.co.aiweb.machinelearning.vo.LabelInfo;
import kr.co.aiweb.machinelearning.vo.PredictResult;
import lombok.extern.slf4j.Slf4j;

/**
 * 
 */
@Slf4j
public class TrainModel_MultiLayer extends TrainModel {

	/**
	 * 학습 모델
	 */
	private MultiLayerNetwork model;
	
	/**
	 * FineTuneConfiguration
	 */
//	private FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
//		    .updater(new Nesterovs(0.01, 0.9))  // 옵티마이저
//		    .seed(DatasetConst.SEED.getValue())
//		    .build();
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
	protected TrainModel_MultiLayer(File rootDir) throws Exception {
		super(rootDir);
	}
	
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#initModel()
	 */
	@Override
	protected void initModel() throws Exception {

		super.modelFile = new File(rootDir, "classification_model_multilayer.zip");
		super.labelFile = new File(rootDir, "classification_label_multilayer.json");
		super.preProcessor = new ImagePreProcessingScaler();

		//label 정보 로드
		labelInfo = new LabelInfo(labelFile);
		
		//모델파일이 존재
		if(modelFile.exists() == true) {
			log.info(".............. 저장 학습모델 loading....... ");
			model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
		}
	}
	
	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#updateModel(int)
	 */
	@Override
	protected void updateModel(int nOut) throws Exception {
		
		//저장한 모델이 없는경우 모델 생성
		if(model == null) {
			MultiLayerConfiguration configuration = configuration(nOut);
			model = new MultiLayerNetwork(configuration);
			model.init();
			model.setListeners(new ScoreIterationListener(DatasetConst.EPOCHS.getValue()));
			return;
		}
		
		//기존 모델이 있는경우 업데이트
		MultiLayerConfiguration configure = model.getLayerWiseConfigurations();
		List<org.deeplearning4j.nn.conf.layers.Layer> layers = configure.getConfs()
				.stream()
				.map(c -> c.getLayer())
				.collect(Collectors.toList());
		
		OutputLayer outputLayer = (OutputLayer)layers.get(layers.size() - 1);
		//기존 out 가 동일할경우
		if(nOut == outputLayer.getNOut()) {
			return;
		}
		
		log.info("=================>>> MultiLayerNetwork model update");
		model = new TransferLearning.Builder(model)
				.fineTuneConfiguration(fineTuneConf)
			    .removeLayersFromOutput(1) // 마지막 OutputLayer 제거
			    .addLayer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
			        .nIn(128)   // 이전 레이어의 출력 수
			        .nOut(nOut)  // 새로운 클래스 개수
			        .activation(Activation.SOFTMAX)
			        .weightInit(WeightInit.XAVIER)
			        .build())
			    .build();
		model.setListeners(new ScoreIterationListener(DatasetConst.EPOCHS.getValue()));
		
//		Criteria<Image, Classifications> criteria = Criteria.builder()
//			    .setTypes(Image.class, Classifications.class)
//			    .optModelUrls("https://mlrepo.djl.ai/model/cv/image_classification/mobilenetv2/ai.djl.pytorch/mobilenetv2.zip")
//			    .optTranslator(ImageClassificationTranslator.builder().build())
//			    .optEngine("PyTorch")
//			    .build();		
		
	}

	/**@Override 
	 * @see kr.co.aiweb.machinelearning.trainmodel.TrainModel#fit(java.io.File)
	 */
	@Override
	public void fit(File datasetDir) throws Exception {
		log.info("---------------------------------- dataset create.");
		DataSetIterator datasetIter = super.createDatasetIterator(datasetDir);
		
		log.info("---------------------------------- model update check :: label count = " + labelInfo.getLabelCount());
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
		
		log.info("................. ImagePreProcessingScaler transform");
        preProcessor.transform(features);
		
        // 예측
		log.info("................. model output");
        INDArray output = model.output(features, false);
        
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

    /**MultiLayerNetwork configuration
     * @param nOut
     * @return
     */
    private MultiLayerConfiguration configuration(int nOut) {
    	
    	//new Adam(1e-4)
        // 모델 구성
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(42)
            //.updater(new Adam(0.001))  // ✅ Adam 사용
            .updater(new Adam(1e-4))  // ✅ Adam 사용
            .weightInit(WeightInit.XAVIER)
            .list()
            .layer(0, new ConvolutionLayer.Builder(3, 3)
                .nIn(DatasetConst.CHANNELS.getValue())
                .nOut(32)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(2, new ConvolutionLayer.Builder(3, 3)
                .nOut(64)
                .stride(1, 1)
                .activation(Activation.RELU)
                .build())
            .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                .kernelSize(2, 2)
                .stride(2, 2)
                .build())
            .layer(4, new DenseLayer.Builder()
                .nOut(128)
                .activation(Activation.RELU)
                .build())
            .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nOut(nOut)
                .activation(Activation.SOFTMAX)
                .build())
            .setInputType(InputType.convolutional(DatasetConst.HEIGHT.getValue(), DatasetConst.WIDTH.getValue(), DatasetConst.CHANNELS.getValue()))
            .build();
    	return conf;
    }	
}
