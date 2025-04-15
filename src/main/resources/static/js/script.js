document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const previewImage = document.getElementById("previewImage");
  const classifyBtn = document.getElementById("classifyBtn");
  const resultBox = document.getElementById("resultBox");
  const labelList = document.getElementById("labelList");

  // 📌 Load label list on page load
  loadLableInfo();
  console.log('111');
  function loadLableInfo() {
	console.log('222');
	fetch('/classification/labels')
	  .then(res => res.json())
	  .then(result => {
		if(result.resultCode != '0000') {
		  alert(result.resultMsg);
		}
		else {
		  labels = result.labelInfo.labelsNameList;
		  labelList.innerHTML = labels.map(label => `<div class="label">${label}</div>`).join('');
		  setTimeout(loadLableInfo, 10000);
		}
	  })
	  .catch(err => {
	    console.error("Failed to load labels:", err);
	    labelList.innerHTML = "<div>Error loading labels</div>";
	  });
  }	

  // 📸 Preview selected image
  imageInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      previewImage.src = reader.result;
	  document.getElementById("previewContainer").style.display = "block"; // 이미지 업로드되면 표시
    };
    reader.readAsDataURL(file);
  });

  // 🚀 Classify image
  classifyBtn.addEventListener("click", () => {
    const file = imageInput.files[0];
    if (!file) {
      alert("Please select an image!");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    fetch('/classification/classify', {
      method: 'POST',
      body: formData
    })
    .then(res => res.json())
    .then(result => {
		if(result.resultCode != '0000') {
		  alert(result.resultMsg);
		  //resultBox.innerText = '';
		  //resultBox.innerHTML = `<b>@ [예측 결과] : <br/> @ [정확도] : `;
		  resultBoxSet('', '');
		}
		else {
		  predictResult = result.predictResult;
		  //resultBox.innerHTML = `<b>@ [예측 결과]</b> : ${predictResult.labelName} <br/> @ [정확도] : ${predictResult.confidence}`;
		  resultBoxSet(predictResult.labelName, predictResult.confidence);
		  const audio = new Audio('/classification/classifyVoice?filename=');
		  audio.play();
		}
    })
    .catch(err => {
      console.error("Classification failed:", err);
      resultBox.innerText = "Error classifying image.";
    });
  });
  
  function resultBoxSet(resultvalue, confidence) {
	resultBox.innerHTML = `<b>@ [예측 결과]</b> : ${resultvalue} <br/> <b>@ [정확도] :</b> ${confidence}`;
  }
  resultBoxSet('', '');
  
  
  
  
  //------------------------
  // 레이어 열기
  //------------------------
  document.getElementById('openTrainLayerBtn')?.addEventListener('click', () => {
	
	fetch('/classification/istraining', {
	  method: 'POST'
	})
	.then(res => res.json())
	.then(result => {
		if(result.resultCode != '0000') {
		  alert(result.resultMsg);
		}
		else {
		  document.getElementById('trainLayer').style.display = 'flex';
		}
	})
	.catch(err => {
	  console.error("Classification failed:", err);
	});
  });

  
  
  //------------------------
  // 레이어 닫기
  //------------------------
  document.getElementById("closeTrainLayerBtn").addEventListener("click", () => {
    document.getElementById("trainLayer").style.display = "none";
  });  

  //------------------------
  // 학습 요청
  //------------------------
  document.getElementById('startTrainBtn').addEventListener('click', async () => {
    const password = document.getElementById('adminPassword').value;
    const zipFile = document.getElementById('zipFileUpload').files[0];

    if (!password || !zipFile) {
      //statusBox.textContent = "패스워드와 ZIP 파일을 모두 입력하세요.";
	  alert('패스워드, 데이터셋파일(ZIP) 모두 입력하세요.');
      return;
    }
	checkTraing();
  });
  
  
  function checkTraing() {
	const password = document.getElementById('adminPassword').value;
	const formData = new FormData();
	formData.append("password", password);
	
	fetch('/classification/checktraining', {
	  method: 'POST',
	  body: formData
	})
	.then(res => res.json())
	.then(result => {
		if(result.resultCode != '0000') {
		  alert(result.resultMsg);
		}
		else {
		  training();
		}
	})
	.catch(err => {
	  console.error("checktraining failed:", err);
	});
  }
  
  function training() {
	const password = document.getElementById('adminPassword').value;
	const zipFile = document.getElementById('zipFileUpload').files[0];
	
	const formData = new FormData();
	formData.append("password", password);
	formData.append("zipFile", zipFile);

	fetch('/classification/training', {
	  method: 'POST',
	  body: formData
	})
	.then(res => res.json())
	.then(result => {
		alert(result.resultMsg);
	})
	.catch(err => {
	  console.error("training failed:", err);
	});
  }
  
  
  
  
  
  
  
  // zip 파일 이름 표시
  const zipInput = document.getElementById("zipFileUpload");
  const zipFileName = document.getElementById("zipFileName");

  zipInput.addEventListener("change", () => {
    if (zipInput.files.length > 0) {
      zipFileName.textContent = zipInput.files[0].name;
    } else {
      zipFileName.textContent = "선택된 파일 없음";
    }
  });
  
  
});
