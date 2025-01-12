#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define IMAGE_ROWS 28
#define IMAGE_COLS 28
#define IMAGE_TRAIN_TIME 350
#define NUM_NEURONS 400  // 예시로 총 뉴런 수를 400개로 설정
#define NUM_SYNAPSES 784 // 각 뉴런이 가지는 시냅스 수 28*28
#define MAX_BITS 15  // 최대 비트수 (15부터 시작)

typedef struct {
    int image_train_time;
    int inhibitory_potential;
    float min_frequency;
    float max_frequency;
    float weight1;
    float weight2;
    float weight3;
    float weight4;
    float spike_threshold;  // 스파이크 발생 임계값
    float resting_potential; // 휴지 막전위
    float spike_drop_rate;  // 스파이크 후 전위 감소율
    float threshold_drop_rate; // 임계값 감소율
    float hyperpolarization_potential;
} Parameters;

typedef struct {
    float potential;  // 현재 막전위
    float adaptive_spike_threshold;  // 적응형 스파이크 임계값
    float refractory_period; // 불응기
    int rest_until;  // 불응기가 끝나는 시간
} Neuron;

//비트, 정확도
typedef struct {
    int bit;
    float accuracy;
} BitAccuracy;

// Parameters 초기화 함수
Parameters initialize_parameters() {
    Parameters params;
    params.image_train_time = IMAGE_TRAIN_TIME;
    params.min_frequency = 1.0;
    params.max_frequency = 50.0;

    //params.weight 3비트
    params.weight1 = 0.625;
    params.weight2 = 0.125;
    params.weight3 = -0.125;
    params.weight4 = -0.5;
    params.spike_threshold = -55.0;
    params.resting_potential = -70.0;
    params.hyperpolarization_potential = -90.0;
    params.spike_drop_rate = 0.8;
    params.threshold_drop_rate = 0.4;
    params.inhibitory_potential = -100;
    return params;
}

// Neuron 초기화 함수
Neuron initialize_neuron(Parameters* params) {
    Neuron neuron;
    neuron.potential = params->resting_potential;  // 초기 막전위 설정
    neuron.adaptive_spike_threshold = params->spike_threshold;
    neuron.rest_until = -1;  // 뉴런이 활성화될 수 있는 초기 시간
    neuron.refractory_period = 15;
    return neuron;
}

void hyperpolarization(Neuron* neuron, Parameters* params, int time_step){
    neuron->potential = params->hyperpolarization_potential;
    neuron->rest_until = time_step + neuron->refractory_period;
}

void inhibit(Neuron* neuron, Parameters* params, int time_step){
    neuron->potential = params->inhibitory_potential;
    neuron->rest_until = time_step + neuron->refractory_period;
}

void inhibit_looser_neurons(int* count_spikes, Neuron* neurons, int num_neurons, int time_step, int winner_index, Parameters* params) {
    for (int looser_neuron_index = 0; looser_neuron_index < num_neurons; looser_neuron_index++) {
        // 승자 뉴런은 건너뜀
        if (looser_neuron_index != winner_index) {
            // 스파이크 조건 확인
            if (neurons[looser_neuron_index].potential > neurons[looser_neuron_index].adaptive_spike_threshold) {
                count_spikes[looser_neuron_index] += 1;
            }

            // 억제 메서드 호출
            inhibit(&neurons[looser_neuron_index], params, time_step);
        }
    }
}


int float_to_fixed(float value, int factor) {
    // Q1.factor 형식으로 변환
    if (value < -1.0) value = -1.0;  // 최소값 제한
    if (value > 1.0) value = 1.0;   // 최대값 제한

    return (int)(value * (1 << factor));  // 2^factor로 스케일링
}

float fixed_to_float(int fixed_value, int fractional_bits) {
    return fixed_value / (float)(1 << fractional_bits);
}

float** bitChange(float** synapse, int bit){
    for (int i = 0; i < NUM_NEURONS; i++) {
        for (int j = 0; j < NUM_SYNAPSES; j++){
            synapse[i][j] = float_to_fixed(synapse[i][j], bit);  // Q1.bit 형식으로 변환
            synapse[i][j] = fixed_to_float(synapse[i][j], bit);
        }
    }

    return synapse;
}

float scale_fractional_part(float num, int bit) {
    int integer_part = (int)num;  // 정수 부분 분리
    float fractional_part = num - integer_part;  // 소수 부분 분리

    // 소수 부분만 스케일링
    int scaled_fraction = (int)(fractional_part * (1 << bit));
    float rescaled_fraction = scaled_fraction / (float)(1 << bit);

    // 정수와 소수 다시 결합
    return integer_part + rescaled_fraction;
}

// 수용영역 필터 적용 함수
void apply_receptive_field(unsigned char* image, float* convoluted_image, Parameters* params) {
    float receptive_field[5][5] = {
        {params->weight4, params->weight3, params->weight2, params->weight3, params->weight4},
        {params->weight3, params->weight2, params->weight1, params->weight2, params->weight3},
        {params->weight2, params->weight1, 1.0, params->weight1, params->weight2},
        {params->weight3, params->weight2, params->weight1, params->weight2, params->weight3},
        {params->weight4, params->weight3, params->weight2, params->weight3, params->weight4}
    };
    float max_summation = 0;
    for (int x = 0; x < IMAGE_ROWS; x++) {
        for (int y = 0; y < IMAGE_COLS; y++) {
            float summation = 0.0;
            for (int i = -2; i <= 2; i++) {
                for (int j = -2; j <= 2; j++) {
                    int x_index = x + i;
                    int y_index = y + j;
                    if (x_index >= 0 && x_index < IMAGE_ROWS && y_index >= 0 && y_index < IMAGE_COLS) {
                        summation += (receptive_field[i + 2][j + 2] * image[x_index * IMAGE_COLS + y_index]) / 255.0;
                    }
                }
            }
            convoluted_image[x * IMAGE_COLS + y] = summation;
        }
    }
}

void encode_image_to_spike_train(float* convoluted_image, int** spike_trains, Parameters* params) {
    float min_pixel_value = 1.0; // 최소 픽셀 값
    float max_pixel_value = 0.0; // 최대 픽셀 값

    // 최소 및 최대 픽셀 값 계산
    for (int idx = 0; idx < IMAGE_COLS * IMAGE_ROWS; idx++) {
        if (convoluted_image[idx] < min_pixel_value) {
            min_pixel_value = convoluted_image[idx];
        }
        if (convoluted_image[idx] > max_pixel_value) {
            max_pixel_value = convoluted_image[idx];
        }
    }

    // 스파이크 트레인 생성
    for (int idx = 0; idx < IMAGE_COLS * IMAGE_ROWS; idx++) {
        float pixel_value = convoluted_image[idx];

        // 픽셀 값을 주파수로 변환
        float frequency = (pixel_value - min_pixel_value) / (max_pixel_value - min_pixel_value) * 
                          (params->max_frequency - params->min_frequency) + params->min_frequency;

        // 주파수 값 보정
        if (frequency <= 0.0) {
            frequency = params->min_frequency;
        }

        // 스파이크 간격 계산
        int spike_time_distance = (int)ceil(params->image_train_time / frequency);
        int next_spike_time = spike_time_distance;

        // 각 spike_train을 0으로 초기화
        for (int t = 0; t <= params->image_train_time; t++) {
            spike_trains[idx][t] = 0;
        }

        // 스파이크 시점 설정
        if (pixel_value > 0.0) { // Python과 동일하게 0인 픽셀은 제외
            while (next_spike_time < params->image_train_time) {
                spike_trains[idx][next_spike_time] = 1;
                next_spike_time += spike_time_distance;
            }
        }
    }
}


// 파일의 데이터 크기를 확인하여 항목 수 계산
long get_file_size(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("파일 열기 오류");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    fclose(file);
    return size;
}

// MNIST 데이터 로드 함수 (동적 메모리 할당)
unsigned char** load_mnist_images(const char* filename, int* num_images) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("파일 열기 오류");
        exit(1);
    }

    long file_size = get_file_size(filename);
    int total_pixels = IMAGE_ROWS * IMAGE_COLS;
    *num_images = file_size / (total_pixels * 3); // assuming each pixel is "255," (3 chars)

    // 동적 메모리 할당
    unsigned char** images = (unsigned char**)malloc(*num_images * sizeof(unsigned char*));
    for (int i = 0; i < *num_images; i++) {
        images[i] = (unsigned char*)malloc(total_pixels * sizeof(unsigned char));
    }

    // 데이터 로드
    for (int i = 0; i < *num_images; i++) {
        for (int j = 0; j < total_pixels; j++) {
            fscanf(file, "%hhu,", &images[i][j]);
        }
    }

    fclose(file);
    return images;
}

void calculate_potential(Neuron* neuron, int neuron_index, int** spike_train, float** synapses, int time_step, Parameters* params, int* spike_count, float* current_potentials) {
    if (neuron->rest_until < time_step) {
        for (int i = 0; i < IMAGE_ROWS * IMAGE_COLS; i++) {
            neuron->potential += synapses[neuron_index][i] * spike_train[i][time_step];
        }
        
        if (neuron->potential > params->resting_potential) {
            neuron->potential -= params->spike_drop_rate;
            
            if (neuron->adaptive_spike_threshold > params->spike_threshold) {
                neuron->adaptive_spike_threshold -= params->threshold_drop_rate;
            }
        }
    
        current_potentials[neuron_index] = neuron->potential;
    }
}

float** load_synapses(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("파일 열기 오류");
        exit(1);
    }

    // 동적 메모리 할당
    float** synapse = (float**)malloc(NUM_NEURONS * sizeof(float*));
    for (int i = 0; i < NUM_NEURONS; i++) {
        synapse[i] = (float*)malloc(NUM_SYNAPSES * sizeof(float));
    }

    // 파일에서 데이터 읽어오기
    for (int i = 0; i < NUM_NEURONS; i++) {
        for (int j = 0; j < NUM_SYNAPSES; j++) {
            if (fscanf(file, "%f,", &synapse[i][j]) != 1) {
                fprintf(stderr, "가중치를 읽는 중 오류 발생 at (%d, %d)\n", i, j);
                exit(1);
            } 
        }
    }

    fclose(file);
    return synapse;
}

int* load_labels(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("파일 열기 오류");
        exit(1);
    }

    // 동적 메모리 할당
    int* labels = (int*)malloc(NUM_NEURONS * sizeof(int));
    float temp_label;

    // 파일에서 데이터를 읽어와서 정수로 변환하여 저장
    for (int i = 0; i < NUM_NEURONS; i++) {
        if (fscanf(file, "%f,", &temp_label) != 1) {
            fprintf(stderr, "라벨 읽는 중 오류 발생 at (%d)\n", i);
            exit(1);
        }
        labels[i] = (int)temp_label; // float을 int로 변환
    }

    fclose(file);
    return labels;
}

int* load_answers(const char* filename, int* num_answers) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("파일 열기 오류");
        exit(1);
    }

    // 처음에 10,000개의 요소를 저장할 수 있도록 메모리 할당
    int* answers = (int*)malloc(10000 * sizeof(int));
    int answer, count = 0;

    while (fscanf(file, "%d", &answer) == 1) {
        answers[count++] = answer;

        // 만약 예상보다 더 많은 데이터가 있으면 메모리를 확장
        if (count >= 10000) {
            answers = (int*)realloc(answers, (count + 1000) * sizeof(int));
        }
    }

    fclose(file);

    *num_answers = count;  // 읽어온 데이터 개수 저장
    return answers;
}

int main() {

    // MNIST 데이터 로드
    int num_images;
    unsigned char** test_images = load_mnist_images("./mnist_test_data.csv", &num_images);

    // Synapses 데이터 로드
    float** synapses = load_synapses("./weights.csv");

    // Label 데이터 로드
    int* labels = load_labels("./labels.csv");
    
    // 정답 데이터 로드
    int num_answers;
    int* answers = load_answers("./mnist_test_labels.csv", &num_answers);

    // Parameters 초기화
    Parameters params = initialize_parameters();

    // 뉴런 배열 생성 및 초기화
    Neuron neurons[NUM_NEURONS];
    for (int i = 0; i < NUM_NEURONS; i++) {
        neurons[i] = initialize_neuron(&params);
    }

    // 스파이크 트레인 배열 생성
    int** spike_trains = (int**)malloc(NUM_SYNAPSES * sizeof(int*));
    for (int i = 0; i < NUM_SYNAPSES; i++) {
        spike_trains[i] = (int*)malloc((params.image_train_time + 1) * sizeof(int));
    }

    int bit; // 비트 수

    int correct_predictions = 0; // 정답 개수 추적

    BitAccuracy results[MAX_BITS];  // 비트수와 정확도를 저장하는 구조체 배열
    int result_index = 20;          // 현재 저장 위치 인덱스

    for(bit = 5; bit > 4; bit -= 1)
    {// 전체 이미지 데이터를 순회하여 각 이미지에 대해 막전위를 계산

        synapses = bitChange(synapses, bit);

        for (int img_index = 0; img_index < 1000; img_index++) {
            float convoluted_image[NUM_SYNAPSES];
            apply_receptive_field(test_images[img_index], convoluted_image, &params);
            encode_image_to_spike_train(convoluted_image, spike_trains, &params);
    
            int spike_count[NUM_NEURONS] = {0};  // 스파이크 횟수 배열 초기화
            int count_spikes[NUM_NEURONS] = {0};
            float current_potentials[NUM_NEURONS] = {0};
            int winner_index;

            // 학습 시간 동안 뉴런의 막전위 계산
            for (int time_step = 1; time_step <= params.image_train_time; time_step++) {
                for (int neuron_index = 0; neuron_index < NUM_NEURONS; neuron_index++) {
                    calculate_potential(&neurons[neuron_index], neuron_index, spike_trains, synapses, time_step, &params, spike_count, current_potentials);
                }
            
                int max_spike_index = 0;
                for (int i = 1; i < NUM_NEURONS; i++) {
                    if (current_potentials[i] > current_potentials[max_spike_index]) {
                        max_spike_index = i;
                    }
                }

                winner_index = max_spike_index;

                if(current_potentials[winner_index] < neurons[winner_index].adaptive_spike_threshold){
                    continue;
                } 

                count_spikes[winner_index] += 1;
                hyperpolarization(&neurons[winner_index], &params, time_step);
                
                neurons[winner_index].adaptive_spike_threshold += 1;
                
                inhibit_looser_neurons(count_spikes, neurons, NUM_NEURONS, time_step, winner_index, &params);
            }
            // int predicted_label = labels[winner_index];
            int max_index = 0;
            for (int i = 1; i < NUM_NEURONS; i++) {
                if (count_spikes[i] > count_spikes[max_index]) {
                    max_index = i;
                }
            }

            int prediction = count_spikes[max_index];
            int predicted_label = labels[max_index];
            int actual_label = answers[img_index];

            // 예측이 맞았는지 확인하고, 맞았으면 correct_predictions 증가
            if (predicted_label == actual_label) {
                correct_predictions++;
            }

            printf("이미지 %d: 예측 라벨: %d, 정답: %d\n", img_index, predicted_label, actual_label);

            // 뉴런 초기화
            for (int i = 0; i < NUM_NEURONS; i++) {
                neurons[i] = initialize_neuron(&params);
            }
        }

        // 정확도 계산 및 출력
        float accuracy = ((float)correct_predictions / 1000) * 100;
        printf("비트: %d, 정확도: %.2f%% (%d/%d)\n", bit, accuracy, correct_predictions, 1000);
        // printf("정확도: %.2f%% (%d/%d)\n", accuracy, correct_predictions, 1000);
        results[bit].bit = bit;
        results[bit].accuracy = accuracy;
        correct_predictions = 0;
    }

    // 반복문 종료 후 결과를 파일에 저장
    // FILE* file = fopen("bit_accuracy_results.csv", "w");
    // if (!file) {
    //     perror("결과 파일 열기 실패");
    //     return 1;
    // }

    // fprintf(file, "Bit, Accuracy\n");  // CSV 헤더 작성
    // for (int i = 0; i < result_index; i++) {
    //     fprintf(file, "%d, %.2f\n", results[i].bit, results[i].accuracy);
    // }
    // fclose(file);
    // printf("결과가 bit_accuracy_results.csv 파일에 저장되었습니다.\n");

    // 메모리 해제
    for (int i = 0; i < NUM_SYNAPSES; i++) {
        free(spike_trains[i]);
    }
    free(spike_trains);

    for (int i = 0; i < num_images; i++) {
        free(test_images[i]);
    }
    free(test_images);

    for (int i = 0; i < NUM_NEURONS; i++) {
        free(synapses[i]);
    }
    free(synapses);

    free(labels);
    free(answers);

    return 0;
}
