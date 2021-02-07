use rand::distributions::{Normal, Beta, Distribution};
use rand::thread_rng;
use rand::Rng;

fn main() {
    const NUM_OF_PARTICIPANTS: usize = 30;
    let simulator: Simulator = Simulator::new(NUM_OF_PARTICIPANTS);
    simulator.run();
}

#[derive(Copy, Clone)]
pub struct BetaDistrubutionParameters {
    alpha: f64,
    beta: f64,
}

pub struct Duelist {
    ideal_borda_score: f64,
    real_borda_score: f64,
    submission_order: usize,
}

pub struct Simulator {
    num_of_participants: usize,
    ideal_winning_percentages_matrix: Vec<Vec<f64>>,
}

impl Simulator {
    pub fn new(num_of_participants: usize) -> Self
    {
        let winning_percentages_matrix = make_random_matrix_data(num_of_participants, 50.0, 16.0);
        let winning_percentages_matrix_filename = String::from("winning_percentages_matrix.csv");
        Simulator::write_matrix_to_csv(&winning_percentages_matrix, &winning_percentages_matrix_filename).expect("Fail to write winning_percentages_matrix");

        Simulator {
            num_of_participants,
            ideal_winning_percentages_matrix:winning_percentages_matrix,
        }
    }

    pub fn run(&self) {
        let mut duelists:Vec<Duelist> = vec![];
        duelists.reserve(self.num_of_participants);
        for (submission_order, winning_percentages) in self.ideal_winning_percentages_matrix.iter().enumerate() {
            let mut num_of_not_none:f64 = 0.0;
            let mut sum:f64 = 0.0;
            for &winning_percentage in winning_percentages.iter() {
                num_of_not_none += 1.0;
                sum += winning_percentage;
            }
            let ideal_borda_score = sum / num_of_not_none;
            duelists.push(Duelist{ideal_borda_score, submission_order, real_borda_score:0.0});
        }

        println!("ideal_borda_score");
        for duelist in duelists.iter() {
            print!("{}, ",duelist.ideal_borda_score);
        }
        println!("");

        let mut beta_distrubution_parameters = vec![BetaDistrubutionParameters{alpha:1.0, beta:1.0};self.num_of_participants];

        // チューニング対象パラメータ
        let num_of_evaluate = 3;
        // Probability of randomly selecting a participant
        let probability_of_random_select = 0.3;
        let reward_winner = 1.0;
        let reward_loser = 1.0;
        let mut num_of_fights: Vec<usize> = vec![0; self.num_of_participants];

        // 結果
        let mut real_results_table = vec![vec![0;self.num_of_participants];self.num_of_participants];
        let mut real_winning_percentages_matrix = vec![vec![None;self.num_of_participants];self.num_of_participants];


        let mut rng = rand::thread_rng();
        // nが増える＝新たな提出者＆評価(+1は最後の一人提出後の先生評価)
        for n in 2..self.num_of_participants + 1 {
            for _round in 0..num_of_evaluate {
                let (left, _right) = num_of_fights.split_at(n);
                let mut duelist1: Option<usize> = None;
                let mut duelist2: Option<usize> = None;

                // まだ1回もバトルしていない腕なら最優先で選ばれる
                for (i, &v) in left.iter().enumerate() {
                    if v == 0 {
                        if duelist1 == None {
                            duelist1 = Some(i);
                        } else if duelist2 == None {
                            duelist2 = Some(i);
                            break;
                        }
                    }
                }

                // 未対戦の腕が0もしくは1つだけだった場合
                if duelist1 == None || duelist2 == None {

                    // デュエルする2本の腕を選ぶ
                    if rng.gen::<f64>() <= probability_of_random_select {
                        // 完全にランダムで選ぶ場合
                        if duelist1 == None {
                            duelist1 = Some(rng.gen_range(0, n));
                        }
                        while duelist2 == None || duelist1 == duelist2 {
                            duelist2 = Some(rng.gen_range(0, n));
                        }
                    } else {
                        // 過去の情報に基づいて選ぶ場合
                        if duelist1 == None {
                            let mut sample = 0.0;
                            for i in 0..n {
                                let beta = Beta::new(beta_distrubution_parameters[i].alpha, beta_distrubution_parameters[i].beta);
                                let v = beta.sample(&mut thread_rng());
                                if sample < v {
                                    sample = v;
                                    duelist1 = Some(i);
                                }
                            }
                        }
                        while duelist2 == None || duelist1 == duelist2 {
                            let mut sample = 0.0;
                            for i in 0..n {
                                let beta = Beta::new(beta_distrubution_parameters[i].alpha, beta_distrubution_parameters[i].beta);
                                let v = beta.sample(&mut thread_rng());
                                if sample < v {
                                    sample = v;
                                    duelist2 = Some(i);
                                }
                            }
                        }
                    }
                }
                let duelist1 = duelist1.unwrap();
                let duelist2 = duelist2.unwrap();
                num_of_fights[duelist1] += 1;
                num_of_fights[duelist2] += 1;
                println!("duelist: {}, {} ", duelist1, duelist2);

                if rng.gen::<f64>() <= self.ideal_winning_percentages_matrix[duelist1][duelist2] {
                    beta_distrubution_parameters[duelist1].alpha += reward_winner;
                    beta_distrubution_parameters[duelist2].beta += reward_loser;
                    real_results_table[duelist1][duelist2] += 1;
                } else {
                    beta_distrubution_parameters[duelist2].alpha += reward_winner;
                    beta_distrubution_parameters[duelist1].beta += reward_loser;
                    real_results_table[duelist2][duelist1] += 1;
                }
            }
        }
        println!("real_results_table");
        for j in 0..real_results_table.len() {
            for i in 0..real_results_table.len() {
                print!("{}, ", real_results_table[j][i]);
            }
            println!("{}", num_of_fights[j]);
        }
        println!("");
        for j in 0..real_results_table.len() {
            for i in 0..real_results_table.len() {
                let num_of_win = real_results_table[j][i];
                let num_of_lose = real_results_table[i][j];
                if 0 <  num_of_win || 0 < num_of_lose{
                    let winning_percentages = (num_of_win as f64) / ((num_of_win + num_of_lose) as f64);
                    println!("num_of_win:{},num_of_lose:{},winning_percentages:{}",num_of_win,num_of_lose,winning_percentages);
                    real_winning_percentages_matrix[j][i] = Some(winning_percentages);
                    real_winning_percentages_matrix[i][j] = Some(1.0 - winning_percentages);
                }
            }
        }
        println!("real_winning_percentages_matrix");
        for j in 0..real_winning_percentages_matrix.len() {
            for i in 0..real_winning_percentages_matrix.len() {
                if real_winning_percentages_matrix[j][i] == None {
                    print!("{}, ", 0);
                } else {
                    print!("{}, ", real_winning_percentages_matrix[j][i].unwrap());
                }
            }
            println!("");
        }


        for (submission_order, winning_percentages) in real_winning_percentages_matrix.iter().enumerate() {
            let mut num_of_not_none:f64 = 0.0;
            let mut sum:f64 = 0.0;
            for &winning_percentage in winning_percentages.iter() {
                if winning_percentage.is_some() {
                    num_of_not_none += 1.0;
                    sum += winning_percentage.unwrap();
                }
            }
            duelists[submission_order].real_borda_score = sum / num_of_not_none;
        }
        println!("real_borda_score");
        for duelist in duelists.iter() {
            print!("{}, ",duelist.real_borda_score);
        }
        println!("");
        for duelist in duelists.iter() {
            println!("{},{},{}",duelist.submission_order, duelist.ideal_borda_score,duelist.ideal_borda_score);
        }


        duelists.sort_by(|a, b| b.ideal_borda_score.partial_cmp(&a.ideal_borda_score).unwrap());
        let mut iDCG = duelists.first().unwrap().ideal_borda_score;
        for i in 2..duelists.len() {
            iDCG += duelists[i - 1].ideal_borda_score / (i as f64).log2();
        }
        println!("iDCG={}",iDCG);
        let mut rDCG = duelists.first().unwrap().real_borda_score;
        for i in 2..duelists.len() {
            rDCG += duelists[i - 1].real_borda_score / (i as f64).log2();
        }
        println!("rDCG={}",rDCG);
        let nDCG = rDCG / iDCG;
        println!("nDCG={}",nDCG);
    }

    fn write_matrix_to_csv(matrix: &Vec<Vec<f64>>, filename: &String) -> Result<(), std::io::Error> {
        let mut wtr = csv::Writer::from_path(&filename)?;
        for values in matrix.iter() {
            wtr.write_record(values.iter().map(|x| x.to_string()))?;
        }
        wtr.flush()?;
        Ok(())
    }
}

pub fn make_random_matrix_data(size: usize, mean: f64, stddev: f64) ->Vec<Vec<f64>>
{
    let mut scores = vec![0.0;size];

    let normal = Normal::new(mean, stddev);
    for score in scores.iter_mut() {
        *score = normal.sample(&mut rand::thread_rng());
    }

    let mut winning_percentages_matrix = vec![vec![0.5;size];size];

    for j in 0..size {
        for i in 0..j {
            if i != j {
                let winning_percentage = (50.0 + (scores[i] - scores[j])) / 100.0;
                winning_percentages_matrix[i][j] = num::clamp(winning_percentage,0.0, 1.0);
                winning_percentages_matrix[j][i] = 1.0 - winning_percentages_matrix[i][j];
            }
        }
    }
    for winning_percentages in winning_percentages_matrix.iter() {
        for winning_percentage in winning_percentages.iter() {
            print!("{}, ",winning_percentage);
        }
        println!();
    }
    winning_percentages_matrix
}