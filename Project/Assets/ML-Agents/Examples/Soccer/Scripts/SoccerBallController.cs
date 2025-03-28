using UnityEngine;

public class SoccerBallController : MonoBehaviour
{
    public GameObject area;
    [HideInInspector]
    public SoccerEnvController envController;
    public string purpleGoalTag; //will be used to check if collided with purple goal
    public string blueGoalTag; //will be used to check if collided with blue goal

    void Start()
    {
        envController = area.GetComponent<SoccerEnvController>();
    }

    void OnCollisionEnter(Collision col)
    {
        if (col.gameObject.CompareTag(purpleGoalTag)) //ball touched purple goal
        {
            envController.GoalTouched(Team.Blue);
        }
        if (col.gameObject.CompareTag(blueGoalTag)) //ball touched blue goal
        {
            envController.GoalTouched(Team.Purple);
        }
    }
    // void OnTriggerEnter(Collider other)
    // {
    //     // ボールが壁に当たった場合の処理
    //     if (other.CompareTag("wall"))
    //     {
    //         Debug.Log("ボールが壁に当たりました（Trigger）。リセットします。");
    //         envController.ResetScene();
    //     }
    // }
}
