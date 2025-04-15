/**
 * 
 */
package kr.co.aiweb.machinelearning.vo;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

import lombok.Getter;

/**
 * 
 */
public class LabelInfo {

	private Map<String, Integer> labelMap;
	
	@Getter
	private List<String> labelsNameList;
	
	private File labelFile;
	
	private ObjectMapper mapper = new ObjectMapper();
	
	
	public LabelInfo(File labelFile) throws Exception {
		this.labelFile = labelFile;
		
		//label 파일 미존재
		if(labelFile.exists() == false) {
			labelMap = new HashMap<>();
			labelsNameList = new ArrayList<>();
			return;
		}
		labelMap = mapper.readValue(labelFile, mapper.getTypeFactory().constructMapType(Map.class, String.class, Integer.class));
		resetLabelsNameList();
	}
	
	/**
	 * labelNamesList 재정의
	 */
	private void resetLabelsNameList() {
		// 인덱스 기준으로 정렬된 라벨 리스트 만들기
        String[] labels = new String[labelMap.size()];

        for (Map.Entry<String, Integer> entry : labelMap.entrySet()) {
            labels[entry.getValue()] = entry.getKey();
        }
        labelsNameList = new ArrayList<>(Arrays.asList(labels));
	}
	
	/**
	 * @param labelName
	 */
	public void putLabel(String labelName) {
		//이미 존재하는 label 이면 아무처리도 하지 않음.
		if(labelMap.containsKey(labelName) == true) {
			return;
		}
		labelMap.put(labelName, labelMap.size());
		labelsNameList.add(labelName);
	}
	
	/**label index
	 * @param labelName
	 * @return
	 */
	public int getLabelIndex(String labelName) {
		return labelMap.get(labelName);
	}
	
	/**label name 취득
	 * @param idx
	 * @return
	 */
	public String getLabelName(int idx) {
		return labelsNameList.get(idx);
	}
	
	/**label 갯수
	 * @return
	 */
	public int getLabelCount() {
		return labelMap.size();
	}
	
	/**저장
	 * @throws Exception
	 */
	public void save() throws Exception {
		mapper.writerWithDefaultPrettyPrinter().writeValue(labelFile, labelMap);
	}
	
}
