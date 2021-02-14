use rand::distributions::{Normal, Beta, Distribution, Open01};
use rand::thread_rng;
use rand::Rng;
use rayon::prelude::*;
use array_macro::*;
use chrono::{Utc, Local, DateTime, Date};

#[derive(Copy, Clone)]
struct Duelist {
    submission_order: usize,
    ideal_borda_score: f64,
    real_borda_score: f64,
    alpha: f64,
    beta: f64,
}

impl Duelist {
    pub fn new(submission_order: usize) -> Self{
        Duelist{submission_order, ideal_borda_score:0.0, real_borda_score:0.0, alpha:1.0, beta:1.0}
    }
    pub fn generate(num_of_participants: usize) -> Vec<Duelist>{
        let mut duelists:Vec<Duelist> = vec![];
        duelists.reserve(num_of_participants);
        for n in 0..num_of_participants{
            duelists.push(Duelist::new(n));
        }
        duelists
    }
}

#[derive(Clone)]
struct SimulationResult {
    time: String,
    num_of_evaluate: i32,
    probability_of_random_select: f64,
    reward_winner: f64,
    reward_loser: f64,
    nDCG: f64,
}

impl SimulationResult {
    pub fn new(time: impl Into<String>, num_of_evaluate: i32, probability_of_random_select: f64,
               reward_winner: f64, reward_loser: f64, nDCG: f64,) -> SimulationResult {
        SimulationResult { time: time.into(),num_of_evaluate,probability_of_random_select, reward_winner,reward_loser,nDCG }
    }
    pub fn print(&self) {
        println!("{}, 1人の評価数={}, ランダム対戦の割合={}, 勝者の報酬={}, 敗者の報酬={}, nDCG={}", self.time, self.num_of_evaluate, self.probability_of_random_select, self.reward_winner, self.reward_loser, self.nDCG);
    }
    pub fn toHeaderString() -> [&'static str;6]  {
        return ["時刻","1人の評価数","ランダム対戦の割合","勝者の報酬","敗者の報酬","nDCG"];
    }
    pub fn toStrings(&self) -> [String; 6] {
        return [self.time.to_string(),
            self.num_of_evaluate.to_string(),
            self.probability_of_random_select.to_string(),
            self.reward_winner.to_string(),
            self.reward_loser.to_string(),
            self.nDCG.to_string()];
    }
}


fn calc_borda_score(winning_percentages: &Vec<f64>) -> f64 {
    let mut num_of_not_none:f64 = 0.0;
    let mut sum:f64 = 0.0;
    for &winning_percentage in winning_percentages.iter() {
        num_of_not_none += 1.0;
        sum += winning_percentage;
    }
    sum / num_of_not_none
}

fn main() {
    // 定数
    const IDEAL_WINNING_PERCENTAGES_MATRIX_FILENAME: &str = "ideal_winning_percentages_matrix.csv";
    const REAL_WINNING_PERCENTAGES_MATRIX_FILENAME:&str = "real_winning_percentages_matrix.csv";
    const RESULTS_FILENAME:&str = "resutls.csv";
    const NUM_OF_PARTICIPANTS: usize = 30;
    const NUM_OF_SIMURATION: usize = 100;

    // 総合結果
    let mut results: Vec<SimulationResult> = vec![];

    let mut wtr = csv::Writer::from_path(RESULTS_FILENAME).expect("Fail to open results file");;
    wtr.write_record(SimulationResult::toHeaderString().iter().map(|x| x.to_string())).expect("Fail to write results");;

    // チューニング対象パラメータ
    for num_of_evaluate in 1..5 {
        for _probability_of_random_select in 0..11 {
            for _reward_winner in 1..41 {
                for _reward_loser in 1..41 {
                    let probability_of_random_select = _probability_of_random_select as f64 / 10.0;
                    let reward_winner = _reward_winner as f64 / 10.0;
                    let reward_loser = _reward_loser as f64 / 10.0;

                    let mut nDCGs: Vec<f64> = vec![];
                    nDCGs.reserve(NUM_OF_SIMURATION);
                    for _simulation in 0..NUM_OF_SIMURATION {
                        // シミュレーション用データ
                        let ideal_winning_percentages_matrix = make_random_matrix_data(NUM_OF_PARTICIPANTS, 50.0, 16.0);
                        let mut real_winning_percentages_matrix = vec![vec![None; NUM_OF_PARTICIPANTS]; NUM_OF_PARTICIPANTS];
                        let mut real_duel_results_table = vec![vec![0; NUM_OF_PARTICIPANTS]; NUM_OF_PARTICIPANTS];
                        let mut duelists: Vec<Duelist> = Duelist::generate(NUM_OF_PARTICIPANTS);
                        let mut num_of_fights: Vec<usize> = vec![0; NUM_OF_PARTICIPANTS];

                        for (i, winning_percentages) in ideal_winning_percentages_matrix.iter().enumerate() {
                            duelists[i].ideal_borda_score = calc_borda_score(winning_percentages);
                        }

                        // println!("ideal_borda_score");
                        // for duelist in duelists.iter() {
                        //     print!("{}, ",duelist.ideal_borda_score);
                        // }
                        // println!();

                        // 結果

                        let mut rng = rand::thread_rng();
                        // nが増える＝新たな提出者＆評価(+1は最後の一人提出後の先生評価)
                        for n in 2..NUM_OF_PARTICIPANTS + 1 {
                            let mut matchups_table = vec![vec![false; n]; n];
                            for i in 0..n {
                                matchups_table[i][i] = true;
                            }
                            // 既存提出者の番号
                            let mut target_list:Vec<usize> = vec![];
                            for i in 0..n {
                                target_list.push(i);
                            }
                            for _round in 0..num_of_evaluate {
                                // ボルダ勝者用ThompsonSampling法で組み合わせを決定
                                let (left, _right) = num_of_fights.split_at(n);
                                let mut duelist1: Option<usize> = None;
                                let mut duelist2: Option<usize> = None;

                                // 未比較の決闘者なら最優先で選ばれる
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

                                let is_use_random = rng.gen::<f64>() <= probability_of_random_select;

                                // 未比較の決闘者が2人居なかった場合
                                if duelist1 == None {
                                    // 比較する2つの決闘者を選ぶ
                                    if is_use_random {
                                        // 完全にランダムで選ぶ場合
                                        duelist1 = Some(target_list[rng.gen_range(0, target_list.len())]);
                                    } else {
                                        // 過去の情報に基づいて選ぶ場合
                                        let mut sample = 0.0;
                                        for &i in target_list.iter() {
                                            let beta = Beta::new(duelists[i].alpha, duelists[i].beta);
                                            let v = beta.sample(&mut thread_rng());
                                            if sample < v {
                                                sample = v;
                                                duelist1 = Some(i);
                                            }
                                        }
                                    }
                                }

                                if duelist2 == None {
                                    let mut target_list2 = target_list.clone();
                                    target_list2.retain(|&x| x != duelist1.unwrap());
                                    // 比較する2つの決闘者を選ぶ
                                    if is_use_random {
                                        // 完全にランダムで選ぶ場合
                                        duelist2 = Some(target_list2[rng.gen_range(0, target_list2.len())]);
                                    } else {
                                        // 過去の情報に基づいて選ぶ場合
                                        let mut sample = 0.0;
                                        for &i in target_list2.iter() {
                                            let beta = Beta::new(duelists[i].alpha, duelists[i].beta);
                                            let v = beta.sample(&mut thread_rng());
                                            if sample < v {
                                                sample = v;
                                                duelist2 = Some(i);
                                            }
                                        }
                                    }
                                }

                                // 決定した組み合わせで比較実施
                                let duelist1 = duelist1.unwrap();
                                let duelist2 = duelist2.unwrap();
                                matchups_table[duelist1][duelist2] = true;
                                matchups_table[duelist2][duelist1] = true;
                                num_of_fights[duelist1] += 1;
                                num_of_fights[duelist2] += 1;

                                // println!("duelist: {}, {} ", duelist1, duelist2);

                                if rng.gen::<f64>() <= ideal_winning_percentages_matrix[duelist1][duelist2] {
                                    duelists[duelist1].alpha += reward_winner;
                                    duelists[duelist2].beta += reward_loser;
                                    real_duel_results_table[duelist1][duelist2] += 1;
                                } else {
                                    duelists[duelist2].alpha += reward_winner;
                                    duelists[duelist1].beta += reward_loser;
                                    real_duel_results_table[duelist2][duelist1] += 1;
                                }

                                //他の全員と対戦済みのduelistは候補から除外
                                for (i,matchups) in matchups_table.iter().enumerate() {
                                    if matchups.iter().all(|&b| b) {
                                        target_list.retain(|&x| x != i);
                                    }
                                }
                                if target_list.len() <= 1 {
                                    break;
                                }
                            }
                        }

                        // シミュレーション結果確認
                        // println!("real_results_table");
                        // for j in 0..real_duel_results_table.len() {
                        //     for i in 0..real_duel_results_table.len() {
                        //         print!("{}, ", real_duel_results_table[j][i]);
                        //     }
                        //     println!("{}", num_of_fights[j]);
                        // }
                        // println!();
                        for j in 0..real_duel_results_table.len() {
                            for i in 0..real_duel_results_table.len() {
                                let num_of_win = real_duel_results_table[j][i];
                                let num_of_lose = real_duel_results_table[i][j];
                                if 0 < num_of_win || 0 < num_of_lose {
                                    let winning_percentages = (num_of_win as f64) / ((num_of_win + num_of_lose) as f64);
                                    // println!("num_of_win:{},num_of_lose:{},winning_percentages:{}", num_of_win, num_of_lose, winning_percentages);
                                    real_winning_percentages_matrix[j][i] = Some(winning_percentages);
                                    real_winning_percentages_matrix[i][j] = Some(1.0 - winning_percentages);
                                }
                            }
                        }
                        // println!("real_winning_percentages_matrix");
                        // for j in 0..real_winning_percentages_matrix.len() {
                        //     for i in 0..real_winning_percentages_matrix.len() {
                        //         if real_winning_percentages_matrix[j][i] == None {
                        //             print!("{}, ", 0);
                        //         } else {
                        //             print!("{}, ", real_winning_percentages_matrix[j][i].unwrap());
                        //         }
                        //     }
                        //     println!();
                        // }


                        for (submission_order, winning_percentages) in real_winning_percentages_matrix.iter().enumerate() {
                            duelists[submission_order].real_borda_score = calc_borda_score(&winning_percentages.iter().filter(|x| x.is_some()).map(|v| v.unwrap()).collect())
                        }
                        // println!("real_borda_score");
                        // for duelist in duelists.iter() {
                        //     print!("{}, ", duelist.real_borda_score);
                        // }
                        // println!();

                        duelists.sort_by(|a, b| b.real_borda_score.partial_cmp(&a.real_borda_score).unwrap());
                        let rDCG = calc_DCG(&duelists);

                        duelists.sort_by(|a, b| b.ideal_borda_score.partial_cmp(&a.ideal_borda_score).unwrap());
                        let iDCG = calc_DCG(&duelists);

                        let nDCG = rDCG / iDCG;
                        // println!("rDCG={}, iDCG={}, nDCG={}", rDCG, iDCG, nDCG);
                        //
                        // println!("#,ideal_borda_score,real_borda_score");
                        // for duelist in duelists.iter() {
                        //     println!("{},{},{}", duelist.submission_order, duelist.ideal_borda_score, duelist.real_borda_score);
                        //}
                        nDCGs.push(nDCG);

                        //write_matrix_to_csv(&ideal_winning_percentages_matrix, IDEAL_WINNING_PERCENTAGES_MATRIX_FILENAME).expect("Fail to write ideal_winning_percentages_matrix");
                        //write_matrix_to_csv(&real_winning_percentages_matrix, REAL_WINNING_PERCENTAGES_MATRIX_FILENAME).expect("Fail to write real_winning_percentages_matrix");
                    }
                    let average_nDCG: f64 = nDCGs.iter().sum::<f64>() / (nDCGs.len() as f64);
                    // println!("average_nDCG={}", average_nDCG);
                    let time = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
                    let result = SimulationResult::new(time, num_of_evaluate, probability_of_random_select, reward_winner, reward_loser, average_nDCG);
                    results.push(result);
                    results.last().unwrap().print();
                    wtr.write_record(&results.last().unwrap().toStrings()).expect("Fail to write result");
                    wtr.flush().expect("Fail to flush results");
                }
            }
        }
    }
    wtr.flush().expect("Fail to flush results");

    println!("シミュレーション終了。結果をソートし出力。");
    results.sort_by(|a, b| b.nDCG.partial_cmp(&a.nDCG).unwrap());
    for result in results.iter() {
        result.print();
    }
    //write_results_to_csv(&results, RESULTS_FILENAME).expect("Fail to write results");
}

fn calc_DCG(duelists: &Vec<Duelist>) -> f64 {
    calc_DCG2(duelists)
}

// そこそこの適合度の要素に対する予測優先
fn calc_DCG1(duelists: &Vec<Duelist>) -> f64 {
    let mut DCG = duelists.first().unwrap().ideal_borda_score;
    for i in 2..duelists.len() {
        let rel_i = duelists[i - 1].ideal_borda_score;
        DCG +=  rel_i / (i as f64).log2();
    }
    DCG
}

// 高い適合度の要素に対する予測優先
fn calc_DCG2(duelists: &Vec<Duelist>) -> f64 {
    let mut DCG:f64 = 0.0;
    for i in 1..duelists.len() {
        let rel_i = duelists[i - 1].ideal_borda_score * 10.0;
        DCG += (2.0_f64.powf(rel_i) - 1.0) / (i as f64 + 1.0).log2();
    }
    DCG
}

fn write_matrix_to_csv(matrix: &Vec<Vec<f64>>, filename: &str) -> Result<(), std::io::Error> {
    let mut wtr = csv::Writer::from_path(&filename)?;
    for values in matrix.iter() {
        wtr.write_record(values.iter().map(|x| x.to_string()))?;
    }
    wtr.flush()?;
    Ok(())
}

fn write_results_to_csv(results: &Vec<SimulationResult>, filename: &str) -> Result<(), std::io::Error> {
    let mut wtr = csv::Writer::from_path(&filename)?;
    wtr.write_record(SimulationResult::toHeaderString().iter().map(|x| x.to_string()))?;
    for result in results.iter() {
        wtr.write_record(&[result.num_of_evaluate.to_string(),result.probability_of_random_select.to_string(), result.reward_winner.to_string(), result.reward_winner.to_string(), result.nDCG.to_string()])?;
    }
    wtr.flush()?;
    Ok(())
}

fn make_random_matrix_data(size: usize, mean: f64, stddev: f64) ->Vec<Vec<f64>>
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
    // for winning_percentages in winning_percentages_matrix.iter() {
    //     for winning_percentage in winning_percentages.iter() {
    //         print!("{}, ",winning_percentage);
    //     }
    //     println!();
    // }
    winning_percentages_matrix
}

// n個からk個を取り出す組み合わせの数
pub fn binom_knuth(n: i32, k: i32) -> i32 {
    (0..n + 1)
        .rev()
        .zip(1..k + 1)
        .fold(1, |mut r, (n, d)| { r *= n; r /= d; r })
}