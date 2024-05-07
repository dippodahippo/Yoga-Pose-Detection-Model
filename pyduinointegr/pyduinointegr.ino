const int led = 13;

void setup() {
  Serial.begin(9600);
  pinMode(led, OUTPUT);

}

void loop() {
  if (Serial.available() > 0) {
    String msg = Serial.readString();

    if (msg == "ON"){
      digitalWrite(led, HIGH);
    }

    else if (msg == "OFF") {
      digitalWrite(led, LOW);
    }

    else {
      digitalWrite(led, LOW);
      for(int i = 0; i < 5; i++){
        digitalWrite(led, HIGH);
        delay(100);
        digitalWrite(led, LOW);
        delay(100);
        }
    }
  }

}