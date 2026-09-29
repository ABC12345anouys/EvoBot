# 参数标定报告（只读，来源 logs/episodes/*.jsonl）
日志文件 760 个，覆盖任务 111 个

## libero_10_00（0/43 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 15.0, 'stop_above': -0.08, 'jit': 0.02, 'timeout_scale': 3.0}
- lift 接触力 F: n=181 P50=40.01N P10=0.00N，F<1N（夹空）占 69/181
- grasp_pt 横向偏移: P50=0.0030m P90=0.0505m
- **建议 `contact_stop_band`: 0.05 → 0.094**（13/136 次 descend 失败时 TCP 停在目标上方 0.066m（P90=0.084），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 10 次：卡住(progress<0.3) 0，慢(progress>0.7) 0，中间 10
- 成功 carry 实测速度 P50=0.0070m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_00")
s.update_params({"contact_stop_band": 0.094})
s.save()
```

## libero_10_01（0/43 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 1.25, 'stop_above': -0.08, 'jit': 0.02, 'timeout_scale': 3.0}
- lift 接触力 F: n=234 P50=2.98N P10=0.00N，F<1N（夹空）占 102/234
- grasp_pt 横向偏移: P50=0.0136m P90=0.0419m
- **建议 `contact_stop_band`: 0.05 → 0.094**（5/44 次 descend 失败时 TCP 停在目标上方 0.073m（P90=0.084），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 22 次：卡住(progress<0.3) 0，慢(progress>0.7) 13，中间 9
- 成功 carry 实测速度 P50=0.0071m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_01")
s.update_params({"contact_stop_band": 0.094})
s.save()
```

## libero_10_02（0/43 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 15.0, 'stop_above': -0.08, 'jit': 0.02, 'timeout_scale': 2.0}
- lift 接触力 F: n=169 P50=0.00N P10=0.00N，F<1N（夹空）占 169/169
- grasp_pt 横向偏移: P50=0.0981m P90=0.1149m
- **建议 `contact_stop_band`: 0.05 → 0.12**（8/107 次 descend 失败时 TCP 停在目标上方 0.139m（P90=0.139），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_02")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_10_03（1/8 成功，主导失败 timeout）
- 当前参数: {'k_descend': 1.15, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.6}
- lift 接触力 F: n=38 P50=12.19N P10=7.76N，F<1N（夹空）占 1/38
- grasp_pt 横向偏移: P50=0.0390m P90=0.0421m
- **建议 `contact_stop_band`: None → 0.061**（10/29 次 descend 失败时 TCP 停在目标上方 0.051m（P90=0.051），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 39 次：卡住(progress<0.3) 0，慢(progress>0.7) 31，中间 8
- 成功 carry 实测速度 P50=0.0036m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_03")
s.update_params({"contact_stop_band": 0.061})
s.save()
```

## libero_10_04（0/44 成功，主导失败 goal_not_reached）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 15.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 3.0}
- lift 接触力 F: n=267 P50=6.29N P10=5.78N，F<1N（夹空）占 2/267
- grasp_pt 横向偏移: P50=0.0440m P90=0.0585m
- **建议 `contact_stop_band`: 0.05 → 0.12**（8/12 次 descend 失败时 TCP 停在目标上方 0.179m（P90=0.179），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0057m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_04")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_10_05（0/35 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.08, 'k_descend': 15.0, 'stop_above': -0.08, 'jit': 0.02, 'timeout_scale': 2.5}
- lift 接触力 F: n=199 P50=32.72N P10=0.00N，F<1N（夹空）占 76/199
- grasp_pt 横向偏移: P50=0.0353m P90=0.0440m
- **建议 `contact_stop_band`: 0.08 → 0.12**（10/83 次 descend 失败时 TCP 停在目标上方 0.158m（P90=0.158），> 豁免带半宽 0.08：OSC 下降受限触发不了接触判定）
- carry 超时 33 次：卡住(progress<0.3) 3，慢(progress>0.7) 18，中间 12
- 3 次 carry 超时但位移 <30%：撞墙/被卡，提 vcap 无效，查 min_clearance/coll_pair 与路径
- 成功 carry 实测速度 P50=0.0078m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_05")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_10_06（0/35 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 5.0314, 'stop_above': 0.0, 'jit': 0.02, 'timeout_scale': 3.0}
- lift 接触力 F: n=224 P50=8.04N P10=5.78N，F<1N（夹空）占 2/224
- grasp_pt 横向偏移: P50=0.0584m P90=0.0585m
- **建议 `contact_stop_band`: 0.05 → 0.06**（1/60 次 descend 失败时 TCP 停在目标上方 0.050m（P90=0.050），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 163 次：卡住(progress<0.3) 0，慢(progress>0.7) 144，中间 19
- 成功 carry 实测速度 P50=0.0134m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_06")
s.update_params({"contact_stop_band": 0.06})
s.save()
```

## libero_10_07（0/35 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 15.0, 'stop_above': -0.08, 'jit': 0.02, 'timeout_scale': 3.0}
- lift 接触力 F: n=237 P50=40.03N P10=0.00N，F<1N（夹空）占 55/237
- grasp_pt 横向偏移: P50=0.0031m P90=0.0381m
- **建议 `contact_stop_band`: 0.05 → 0.083**（25/140 次 descend 失败时 TCP 停在目标上方 0.072m（P90=0.073），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 31 次：卡住(progress<0.3) 0，慢(progress>0.7) 5，中间 26
- 成功 carry 实测速度 P50=0.0091m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_10_07")
s.update_params({"contact_stop_band": 0.083})
s.save()
```

## libero_10_08（0/36 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 15.0, 'stop_above': -0.08, 'jit': 0.02, 'timeout_scale': 2.0}
- lift 接触力 F: n=148 P50=0.00N P10=0.00N，F<1N（夹空）占 148/148
- grasp_pt 横向偏移: P50=0.0892m P90=0.1184m
- 成功 carry 实测速度 P50=0.0082m/步 

## libero_10_09（0/35 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 15.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 3.0}
- lift 接触力 F: n=243 P50=41.12N P10=17.75N，F<1N（夹空）占 0/243
- grasp_pt 横向偏移: P50=0.0365m P90=0.0391m
- carry 超时 147 次：卡住(progress<0.3) 0，慢(progress>0.7) 77，中间 70
- 成功 carry 实测速度 P50=0.0073m/步 

## libero_90_00（2/5 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0418m P90=0.0418m

## libero_90_01（0/2 成功，主导失败 other）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0379m P90=0.0379m

## libero_90_02（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=11.49N P10=11.49N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0418m P90=0.0418m
- 成功 carry 实测速度 P50=0.0028m/步 

## libero_90_03（0/6 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=6 P50=39.46N P10=0.00N，F<1N（夹空）占 1/6
- grasp_pt 横向偏移: P50=0.0088m P90=0.0199m
- **建议 `contact_stop_band`: None → 0.12**（5/5 次 descend 失败时 TCP 停在目标上方 0.149m（P90=0.173），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0046m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_03")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_90_04（0/2 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=2 P50=0.00N P10=0.00N，F<1N（夹空）占 1/2
- grasp_pt 横向偏移: P50=0.0088m P90=0.0104m
- 成功 carry 实测速度 P50=0.0044m/步 

## libero_90_05（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0261m P90=0.0261m
- **建议 `contact_stop_band`: None → 0.12**（6/6 次 descend 失败时 TCP 停在目标上方 0.121m（P90=0.124），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_05")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_90_06（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0388m P90=0.0388m

## libero_90_07（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0387m P90=0.0387m

## libero_90_08（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0398m P90=0.0398m
- **建议 `contact_stop_band`: None → 0.12**（23/24 次 descend 失败时 TCP 停在目标上方 0.169m（P90=0.185），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_08")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_90_09（1/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0411m P90=0.0411m
- 成功 carry 实测速度 P50=0.0050m/步 

## libero_90_10（0/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0379m P90=0.0379m

## libero_90_11（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0375m P90=0.0375m

## libero_90_12（1/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=16.48N P10=16.48N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0367m P90=0.0367m
- 成功 carry 实测速度 P50=0.0036m/步 

## libero_90_13（1/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=16.20N P10=16.20N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0367m P90=0.0367m
- 成功 carry 实测速度 P50=0.0036m/步 

## libero_90_14（1/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=15.47N P10=15.47N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0367m P90=0.0367m
- 成功 carry 实测速度 P50=0.0042m/步 

## libero_90_15（0/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0393m P90=0.0393m

## libero_90_16（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=14.72N P10=14.72N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0375m P90=0.0375m
- 成功 carry 实测速度 P50=0.0029m/步 

## libero_90_17（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=15.46N P10=15.46N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0375m P90=0.0375m
- 成功 carry 实测速度 P50=0.0036m/步 

## libero_90_18（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.2251m P90=0.2251m

## libero_90_19（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0868m P90=0.0868m

## libero_90_20（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.2322m P90=0.2322m

## libero_90_21（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.2383m P90=0.2383m
- **建议 `contact_stop_band`: None → 0.071**（4/7 次 descend 失败时 TCP 停在目标上方 0.060m（P90=0.061），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_21")
s.update_params({"contact_stop_band": 0.071})
s.save()
```

## libero_90_22（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0403m P90=0.0403m

## libero_90_23（0/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0376m P90=0.0376m

## libero_90_24（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=18.62N P10=18.62N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0389m P90=0.0389m
- **建议 `contact_stop_band`: None → 0.083**（4/4 次 descend 失败时 TCP 停在目标上方 0.073m（P90=0.073），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 7 次：卡住(progress<0.3) 0，慢(progress>0.7) 7，中间 0
- 成功 carry 实测速度 P50=0.0040m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_24")
s.update_params({"contact_stop_band": 0.083})
s.save()
```

## libero_90_25（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=7.73N P10=7.73N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0380m P90=0.0380m
- 成功 carry 实测速度 P50=0.0042m/步 

## libero_90_26（1/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0076m P90=0.0076m
- 成功 carry 实测速度 P50=0.0053m/步 

## libero_90_27（1/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0194m P90=0.0194m
- 成功 carry 实测速度 P50=0.0040m/步 

## libero_90_28（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0380m P90=0.0380m

## libero_90_29（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=17.24N P10=17.24N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0380m P90=0.0380m
- 成功 carry 实测速度 P50=0.0032m/步 

## libero_90_30（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=3.98N P10=3.98N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0380m P90=0.0380m
- 成功 carry 实测速度 P50=0.0042m/步 

## libero_90_31（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=17.70N P10=17.70N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0380m P90=0.0380m
- 成功 carry 实测速度 P50=0.0044m/步 

## libero_90_32（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=6.98N P10=6.98N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0327m P90=0.0327m
- 成功 carry 实测速度 P50=0.0044m/步 

## libero_90_33（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0585m P90=0.0585m

## libero_90_34（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0378m P90=0.0378m
- **建议 `contact_stop_band`: None → 0.12**（8/8 次 descend 失败时 TCP 停在目标上方 0.141m（P90=0.141），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_34")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_90_35（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0313m P90=0.0313m

## libero_90_36（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=8.46N P10=8.46N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0313m P90=0.0313m
- 成功 carry 实测速度 P50=0.0059m/步 

## libero_90_37（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=6.96N P10=6.96N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0313m P90=0.0313m
- 成功 carry 实测速度 P50=0.0073m/步 

## libero_90_38（1/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0336m P90=0.0336m
- 成功 carry 实测速度 P50=0.0081m/步 

## libero_90_39（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.1065m P90=0.1065m

## libero_90_40（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.2273m P90=0.2273m
- **建议 `contact_stop_band`: None → 0.112**（23/23 次 descend 失败时 TCP 停在目标上方 0.102m（P90=0.102），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_40")
s.update_params({"contact_stop_band": 0.112})
s.save()
```

## libero_90_41（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.2272m P90=0.2272m
- **建议 `contact_stop_band`: None → 0.112**（23/23 次 descend 失败时 TCP 停在目标上方 0.102m（P90=0.102），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_41")
s.update_params({"contact_stop_band": 0.112})
s.save()
```

## libero_90_42（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.2273m P90=0.2273m
- **建议 `contact_stop_band`: None → 0.112**（23/23 次 descend 失败时 TCP 停在目标上方 0.102m（P90=0.102），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_42")
s.update_params({"contact_stop_band": 0.112})
s.save()
```

## libero_90_43（1/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0299m P90=0.0299m
- 成功 carry 实测速度 P50=0.0043m/步 

## libero_90_44（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0311m P90=0.0311m

## libero_90_45（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.2278m P90=0.2278m
- **建议 `contact_stop_band`: None → 0.112**（24/24 次 descend 失败时 TCP 停在目标上方 0.102m（P90=0.102），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_45")
s.update_params({"contact_stop_band": 0.112})
s.save()
```

## libero_90_46（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=40.01N P10=40.01N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0030m P90=0.0030m
- 成功 carry 实测速度 P50=0.0057m/步 

## libero_90_47（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0154m P90=0.0154m

## libero_90_48（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0280m P90=0.0280m

## libero_90_49（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=41.12N P10=41.12N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0032m P90=0.0032m
- 成功 carry 实测速度 P50=0.0066m/步 

## libero_90_50（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=40.00N P10=40.00N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0031m P90=0.0031m
- 成功 carry 实测速度 P50=0.0064m/步 

## libero_90_51（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0002m P90=0.0002m
- **建议 `contact_stop_band`: None → 0.066**（5/7 次 descend 失败时 TCP 停在目标上方 0.053m（P90=0.056），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_90_51")
s.update_params({"contact_stop_band": 0.066})
s.save()
```

## libero_90_52（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0000m P90=0.0000m

## libero_90_53（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0409m P90=0.0409m

## libero_90_54（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=40.14N P10=40.14N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0032m P90=0.0032m
- 成功 carry 实测速度 P50=0.0047m/步 

## libero_90_55（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=41.17N P10=41.17N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0030m P90=0.0030m
- 成功 carry 实测速度 P50=0.0073m/步 

## libero_90_56（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0200m P90=0.0200m

## libero_90_57（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0239m P90=0.0239m

## libero_90_58（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0372m P90=0.0372m

## libero_90_59（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=40.07N P10=40.07N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0032m P90=0.0032m
- 成功 carry 实测速度 P50=0.0055m/步 

## libero_90_60（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=20.15N P10=20.15N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0389m P90=0.0389m
- 成功 carry 实测速度 P50=0.0063m/步 

## libero_90_61（0/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0207m P90=0.0207m

## libero_90_62（1/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0240m P90=0.0240m
- 成功 carry 实测速度 P50=0.0092m/步 

## libero_90_63（1/1 成功，主导失败 grip_failed）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=20.38N P10=20.38N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0422m P90=0.0422m
- carry 超时 2 次：卡住(progress<0.3) 0，慢(progress>0.7) 2，中间 0
- 成功 carry 实测速度 P50=0.0144m/步 

## libero_90_64（0/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=21.79N P10=21.79N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0369m P90=0.0369m
- carry 超时 2 次：卡住(progress<0.3) 2，慢(progress>0.7) 0，中间 0
- 2 次 carry 超时但位移 <30%：撞墙/被卡，提 vcap 无效，查 min_clearance/coll_pair 与路径
- 成功 carry 实测速度 P50=0.0051m/步 

## libero_90_65（1/1 成功，主导失败 ）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=17.65N P10=17.65N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0429m P90=0.0429m
- 成功 carry 实测速度 P50=0.0080m/步 

## libero_90_66（1/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=13.72N P10=13.72N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0426m P90=0.0426m
- 成功 carry 实测速度 P50=0.0082m/步 

## libero_90_67（0/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=8.74N P10=8.74N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0581m P90=0.0581m
- 成功 carry 实测速度 P50=0.0051m/步 

## libero_90_68（1/1 成功，主导失败 goal_not_reached）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=18.62N P10=18.62N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0365m P90=0.0365m
- 成功 carry 实测速度 P50=0.0056m/步 

## libero_90_69（0/1 成功，主导失败 -）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0261m P90=0.0261m

## libero_90_70（0/1 成功，主导失败 -）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0245m P90=0.0245m

## libero_goal_00（3/3 成功，主导失败 ）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0379m P90=0.0379m

## libero_goal_01（3/3 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 2.7788, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.3742}
- lift 接触力 F: n=3 P50=9.59N P10=4.78N，F<1N（夹空）占 0/3
- grasp_pt 横向偏移: P50=0.0380m P90=0.0380m
- carry 超时 1 次：卡住(progress<0.3) 0，慢(progress>0.7) 1，中间 0
- 成功 carry 实测速度 P50=0.0032m/步 

## libero_goal_02（1/6 成功，主导失败 other）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 4.4623, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 2.0114}
- lift 接触力 F: n=14 P50=14.98N P10=14.98N，F<1N（夹空）占 0/14
- grasp_pt 横向偏移: P50=0.0009m P90=0.0202m
- carry 超时 8 次：卡住(progress<0.3) 0，慢(progress>0.7) 0，中间 8
- 成功 carry 实测速度 P50=0.0005m/步 

## libero_goal_03（1/4 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 1.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 2.9122}
- lift 接触力 F: n=1 P50=8.14N P10=8.14N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0398m P90=0.0420m
- **建议 `contact_stop_band`: 0.05 → 0.12**（20/20 次 descend 失败时 TCP 停在目标上方 0.117m（P90=0.149），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0024m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_goal_03")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_goal_04（1/6 成功，主导失败 other）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 1.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 2.9122}
- lift 接触力 F: n=1 P50=20.26N P10=20.26N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0399m P90=0.0421m
- **建议 `contact_stop_band`: 0.05 → 0.12**（32/32 次 descend 失败时 TCP 停在目标上方 0.113m（P90=0.138），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0028m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_goal_04")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_goal_05（1/24 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.03, 'contact_stop_band': 0.05, 'k_descend': 10.0, 'stop_above': -0.029, 'jit': 0.02, 'timeout_scale': 1.6}
- lift 接触力 F: n=154 P50=0.00N P10=0.00N，F<1N（夹空）占 108/154
- grasp_pt 横向偏移: P50=0.0727m P90=0.0812m
- 成功 carry 实测速度 P50=0.0061m/步 

## libero_goal_06（1/4 成功，主导失败 other）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=10 P50=0.00N P10=0.00N，F<1N（夹空）占 9/10
- grasp_pt 横向偏移: P50=0.0001m P90=0.0172m
- 成功 carry 实测速度 P50=0.0026m/步 

## libero_goal_07（2/2 成功，主导失败 ）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- grasp_pt 横向偏移: P50=0.0379m P90=0.0379m

## libero_goal_08（1/3 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=7 P50=8.89N P10=4.86N，F<1N（夹空）占 0/7
- grasp_pt 横向偏移: P50=0.0380m P90=0.0383m
- 成功 carry 实测速度 P50=0.0006m/步 

## libero_goal_09（1/5 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=11 P50=14.97N P10=14.97N，F<1N（夹空）占 1/11
- grasp_pt 横向偏移: P50=0.0010m P90=0.0171m
- **建议 `contact_stop_band`: 0.05 → 0.118**（24/29 次 descend 失败时 TCP 停在目标上方 0.101m（P90=0.108），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0023m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_goal_09")
s.update_params({"contact_stop_band": 0.118})
s.save()
```

## libero_object_00（3/3 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=3 P50=14.39N P10=0.00N，F<1N（夹空）占 1/3
- grasp_pt 横向偏移: P50=0.0294m P90=0.0294m
- carry 超时 1 次：卡住(progress<0.3) 0，慢(progress>0.7) 1，中间 0
- 成功 carry 实测速度 P50=0.0070m/步 

## libero_object_01（3/8 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 8.3365, 'stop_above': -0.029, 'jit': 0.0195, 'timeout_scale': 1.5804}
- lift 接触力 F: n=22 P50=0.00N P10=0.00N，F<1N（夹空）占 19/22
- grasp_pt 横向偏移: P50=0.0093m P90=0.0213m
- carry 超时 2 次：卡住(progress<0.3) 0，慢(progress>0.7) 0，中间 2
- 成功 carry 实测速度 P50=0.0049m/步 

## libero_object_02（2/6 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 9.775, 'stop_above': -0.0195, 'jit': 0.0, 'timeout_scale': 1.15}
- lift 接触力 F: n=17 P50=0.00N P10=0.00N，F<1N（夹空）占 14/17
- grasp_pt 横向偏移: P50=0.0331m P90=0.0360m
- **建议 `contact_stop_band`: 0.05 → 0.101**（41/41 次 descend 失败时 TCP 停在目标上方 0.056m（P90=0.091），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0055m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_object_02")
s.update_params({"contact_stop_band": 0.101})
s.save()
```

## libero_object_03（2/2 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=2 P50=0.00N P10=0.00N，F<1N（夹空）占 1/2
- grasp_pt 横向偏移: P50=0.0001m P90=0.0181m
- 成功 carry 实测速度 P50=0.0058m/步 

## libero_object_04（4/17 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.15, 'contact_stop_band': 0.05, 'k_descend': 1.1289, 'stop_above': -0.0345, 'jit': 0.01, 'timeout_scale': 3.0}
- lift 接触力 F: n=77 P50=33.72N P10=0.00N，F<1N（夹空）占 16/77
- grasp_pt 横向偏移: P50=0.0284m P90=0.0359m
- **建议 `contact_stop_band`: 0.05 → 0.105**（22/26 次 descend 失败时 TCP 停在目标上方 0.086m（P90=0.095），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 11 次：卡住(progress<0.3) 0，慢(progress>0.7) 10，中间 1
- 成功 carry 实测速度 P50=0.0063m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_object_04")
s.update_params({"contact_stop_band": 0.105})
s.save()
```

## libero_object_05（1/1 成功，主导失败 timeout）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=41.51N P10=41.51N，F<1N（夹空）占 0/1
- grasp_pt 横向偏移: P50=0.0069m P90=0.0069m
- **建议 `contact_stop_band`: None → 0.062**（1/1 次 descend 失败时 TCP 停在目标上方 0.052m（P90=0.052），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0060m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_object_05")
s.update_params({"contact_stop_band": 0.062})
s.save()
```

## libero_object_06（1/1 成功，主导失败 other）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=1 P50=0.00N P10=0.00N，F<1N（夹空）占 1/1
- grasp_pt 横向偏移: P50=0.0210m P90=0.0210m
- 成功 carry 实测速度 P50=0.0061m/步 

## libero_object_07（4/21 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.15, 'contact_stop_band': 0.05, 'k_descend': 1.15, 'stop_above': -0.034, 'jit': 0.0, 'timeout_scale': 3.0}
- lift 接触力 F: n=101 P50=40.00N P10=0.00N，F<1N（夹空）占 15/101
- grasp_pt 横向偏移: P50=0.0000m P90=0.0000m
- carry 超时 11 次：卡住(progress<0.3) 0，慢(progress>0.7) 10，中间 1
- 成功 carry 实测速度 P50=0.0041m/步 

## libero_object_08（1/2 成功，主导失败 other）
- 当前参数: {'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=2 P50=0.00N P10=0.00N，F<1N（夹空）占 2/2
- grasp_pt 横向偏移: P50=0.0183m P90=0.0262m
- 成功 carry 实测速度 P50=0.0063m/步 

## libero_object_09（8/44 成功，主导失败 grip_failed）
- 当前参数: {'carry_vcap': 0.1, 'contact_stop_band': 0.05, 'k_descend': 1.25, 'stop_above': -0.08, 'jit': 0.02, 'timeout_scale': 3.0}
- lift 接触力 F: n=253 P50=0.00N P10=0.00N，F<1N（夹空）占 225/253
- grasp_pt 横向偏移: P50=0.0405m P90=0.0446m
- **建议 `contact_stop_band`: 0.05 → 0.091**（7/27 次 descend 失败时 TCP 停在目标上方 0.061m（P90=0.081），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 21 次：卡住(progress<0.3) 0，慢(progress>0.7) 16，中间 5
- 成功 carry 实测速度 P50=0.0013m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_object_09")
s.update_params({"contact_stop_band": 0.091})
s.save()
```

## libero_spatial_00（104/111 成功，主导失败 ）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 3.0036, 'stop_above': -0.015, 'jit': 0.0, 'timeout_scale': 1.5804}
- lift 接触力 F: n=113 P50=14.75N P10=11.43N，F<1N（夹空）占 6/113
- grasp_pt 横向偏移: P50=0.0274m P90=0.0383m
- **建议 `contact_stop_band`: 0.05 → 0.12**（2/8 次 descend 失败时 TCP 停在目标上方 0.259m（P90=0.259），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0027m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_spatial_00")
s.update_params({"contact_stop_band": 0.12})
s.save()
```

## libero_spatial_01（4/5 成功，主导失败 ）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 4.5156, 'stop_above': 0.0, 'jit': 0.005, 'timeout_scale': 1.3225}
- lift 接触力 F: n=4 P50=19.83N P10=16.96N，F<1N（夹空）占 0/4
- grasp_pt 横向偏移: P50=0.0379m P90=0.0379m
- carry 超时 1 次：卡住(progress<0.3) 0，慢(progress>0.7) 0，中间 1
- 成功 carry 实测速度 P50=0.0045m/步 

## libero_spatial_02（4/6 成功，主导失败 ）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.4274, 'stop_above': 0.0, 'jit': 0.015, 'timeout_scale': 1.3742}
- lift 接触力 F: n=10 P50=10.45N P10=4.95N，F<1N（夹空）占 0/10
- grasp_pt 横向偏移: P50=0.0379m P90=0.0382m
- 成功 carry 实测速度 P50=0.0008m/步 

## libero_spatial_03（4/5 成功，主导失败 ）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.3125, 'stop_above': 0.0, 'jit': 0.005, 'timeout_scale': 1.15}
- lift 接触力 F: n=5 P50=10.69N P10=6.09N，F<1N（夹空）占 0/5
- grasp_pt 横向偏移: P50=0.0379m P90=0.0379m
- 成功 carry 实测速度 P50=0.0006m/步 

## libero_spatial_04（2/5 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 1.5685, 'stop_above': 0.0, 'jit': 0.005, 'timeout_scale': 1.9597}
- lift 接触力 F: n=5 P50=0.00N P10=0.00N，F<1N（夹空）占 3/5
- grasp_pt 横向偏移: P50=0.0394m P90=0.0421m
- **建议 `contact_stop_band`: 0.05 → 0.066**（13/14 次 descend 失败时 TCP 停在目标上方 0.052m（P90=0.056），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- 成功 carry 实测速度 P50=0.0075m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_spatial_04")
s.update_params({"contact_stop_band": 0.066})
s.save()
```

## libero_spatial_05（1/5 成功，主导失败 other）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=10 P50=4.97N P10=0.00N，F<1N（夹空）占 3/10
- grasp_pt 横向偏移: P50=0.0422m P90=0.0454m
- carry 超时 4 次：卡住(progress<0.3) 0，慢(progress>0.7) 1，中间 3
- 成功 carry 实测速度 P50=0.0015m/步 

## libero_spatial_06（1/7 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 1.443, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.6}
- lift 接触力 F: n=11 P50=11.24N P10=6.14N，F<1N（夹空）占 0/11
- grasp_pt 横向偏移: P50=0.0337m P90=0.0382m
- **建议 `contact_stop_band`: 0.05 → 0.108**（35/47 次 descend 失败时 TCP 停在目标上方 0.084m（P90=0.098），> 豁免带半宽 0.05：OSC 下降受限触发不了接触判定）
- carry 超时 16 次：卡住(progress<0.3) 0，慢(progress>0.7) 16，中间 0
- 成功 carry 实测速度 P50=0.0049m/步 
```python
from darwin.skills.skill_config import SkillConfigStore
s = SkillConfigStore.load("ik_servo", "libero", task="libero_spatial_06")
s.update_params({"contact_stop_band": 0.108})
s.save()
```

## libero_spatial_07（2/4 成功，主导失败 timeout）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=5 P50=21.14N P10=18.09N，F<1N（夹空）占 0/5
- grasp_pt 横向偏移: P50=0.0379m P90=0.0381m
- carry 超时 1 次：卡住(progress<0.3) 0，慢(progress>0.7) 0，中间 1
- 成功 carry 实测速度 P50=0.0014m/步 

## libero_spatial_08（3/3 成功，主导失败 ）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=3 P50=12.99N P10=4.65N，F<1N（夹空）占 0/3
- grasp_pt 横向偏移: P50=0.0379m P90=0.0380m
- 成功 carry 实测速度 P50=0.0027m/步 

## libero_spatial_09（1/5 成功，主导失败 goal_not_reached）
- 当前参数: {'carry_vcap': 0.05, 'contact_stop_band': 0.05, 'k_descend': 5.0, 'stop_above': 0.0, 'jit': 0.0, 'timeout_scale': 1.0}
- lift 接触力 F: n=9 P50=8.19N P10=4.98N，F<1N（夹空）占 0/9
- grasp_pt 横向偏移: P50=0.0380m P90=0.0382m
- carry 超时 6 次：卡住(progress<0.3) 0，慢(progress>0.7) 1，中间 5
- 成功 carry 实测速度 P50=0.0082m/步 

