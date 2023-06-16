import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';
import 'package:fruits360/utils/mask_labels.dart';

class MyHomePage extends StatefulWidget {
  const MyHomePage({Key? key, required this.title}) : super(key: key);

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late File pickedImage;
  bool isImageLoaded = false;

  List _result = [];
  String _confidence = "";
  dynamic _name = "";
  String fruits = "";

  getImageFromGallery() async {
    var tempStore = await ImagePicker().pickImage(source: ImageSource.gallery);

    setState(() {
      pickedImage = File(tempStore!.path);
      isImageLoaded = true;
      applyModelOnImage(pickedImage);
    });
  }

  getImageFromCamera() async {
    var tempStore = await ImagePicker().pickImage(source: ImageSource.camera);

    setState(() {
      pickedImage = File(tempStore!.path);
      isImageLoaded = true;
      applyModelOnImage(pickedImage);
    });
  }

  resetImage() {
    setState(() {
      isImageLoaded = false;
      _result = [];
      _confidence = "";
      _name = "";
    });
  }

  loadCnnModel() async {
    var resultant = await Tflite.loadModel(
      model: "assets/model.tflite",
      labels: "assets/labels.txt",
    );

    debugPrint("Result after Loading Model : $resultant");
  }

  applyModelOnImage(File _file) async {
    var res = await Tflite.runModelOnImage(
      path: _file.path,
      numResults: 2,
      threshold: 0.1,
      imageMean: 127.5,
      imageStd: 127.5,
    );
    debugPrint("Result after Applying Model : $res");

    if (res?.isNotEmpty ?? false) {
      setState(() {
        _result = res!;
        debugPrint(_result.toString());
        _name = maskLabels[_result[0]["label"]];
        _confidence =
            (_result[0]["confidence"] * 100).toString().substring(0, 2) + "%";
      });
    }
  }

  @override
  void initState() {
    super.initState();
    loadCnnModel();
  }

  @override
  Widget build(BuildContext context) {
    Size pgSize = MediaQuery.of(context).size;
    return Scaffold(
      appBar: AppBar(
        toolbarHeight: pgSize.height * 0.08,
        title: Text(
          widget.title,
        ),
        backgroundColor: const Color.fromARGB(255, 3, 36, 121),
      ),
      body: Container(
        alignment: Alignment.center,
        padding: const EdgeInsets.all(50),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.start,
          children: <Widget>[
            const Text(
              "Face Mask Detection",
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 35,
                color: Color.fromARGB(255, 15, 81, 157),
              ),
            ),
            SizedBox(height: pgSize.height * 0),
            isImageLoaded
                ? Padding(
                    padding: EdgeInsets.only(top: pgSize.height * 0),
                    child: Container(
                      height: pgSize.height * 0.4,
                      width: pgSize.width * 0.5,
                      alignment: Alignment.center,
                      decoration: BoxDecoration(
                        image: DecorationImage(
                          image: FileImage(File(pickedImage.path)),
                          fit: BoxFit.contain,
                        ),
                      ),
                    ),
                  ) // if no image is loaded, show an empty SizedBox
                : Container(
                    height: pgSize.height * 0.2,
                    width: pgSize.width * 0.5,
                    alignment: Alignment.center,
                    child: ListTile(
                      title: Image.asset(
                        "assets/coronavirus.png",
                        width: pgSize.width * 0.1,
                      ),
                      subtitle: const Text(
                        "Upload an Image",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                          color: Colors.green,
                          fontSize: 18,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ),
            SizedBox(
              height: pgSize.height * 0.0,
            ),
            //display large pop up of mask detected or not detected using _name in set state

            Visibility(
              visible: _name.isNotEmpty,
              child: Container(
                // height: pgSize.height * 0.1,
                width: pgSize.width * 0.75,
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(10),
                  color:  (_name == maskLabels.values.first)
                      ? Colors.green.withOpacity(0.7) 
                      : Colors.red.withOpacity(0.7),
                ),
                alignment: Alignment.center,
                child: ListTile(
                  title: FittedBox(
                    child: Text(
                      _name,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white,
                        // fontSize: 28,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  subtitle: Text(
                    "Confidence : $_confidence",
                    textAlign: TextAlign.center,
                    style: const TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ),
            // Visibility(
            //   visible: _confidence != "",
            //   child: Text(
            //     "Confidence : $_confidence",
            //     style: TextStyle(
            //       fontSize: 24,
            //       fontWeight: FontWeight.bold,
            //     ),
            //   ),
            // ),
          ],
        ),
      ),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(
            width: pgSize.width * 0.08,
          ),
          FloatingActionButton(
            tooltip: 'Upload an Image (gallery)',
            onPressed: () {
              getImageFromGallery();
            },
            child: const Icon(Icons.photo_library),
          ),
          const SizedBox(
            width: 25,
          ),
          FloatingActionButton(
            tooltip: 'Upload an Image (camera)',
            onPressed: () {
              getImageFromCamera();
            },
            child: const Icon(Icons.camera_enhance),
          ),
          const SizedBox(
            width: 25,
          ),
          FloatingActionButton(
            tooltip: 'Reset Image',
            onPressed: () {
              resetImage();
            },
            child: const Icon(Icons.replay_circle_filled),
          ),
        ],
      ),
    );
  }
}
