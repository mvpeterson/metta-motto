!(import! &self motto)
!(import! &self agents)
!(import! &self agents:events)
!(bind! &events (basic-event-bus))

!(bind! &a1 (listening-agent "examples/listening_agent_users/simple_call_users.msa"  &events))
!(&a1 .start)
!(publish-event &events  speechstart (((py-atom datetime.datetime.now)) 1) )
!(publish-event &events speech ((py-dict (("data" "My name is Adele") )) 1) )
!((py-atom time.sleep) 5)
!(&a1 .say)

!(publish-event &events  speechstart (((py-atom datetime.datetime.now)) 2) )
!(publish-event &events speech ((py-dict (("data" "My name is Kirk"))) 2))
!((py-atom time.sleep) 5)
!(&a1 .say)

!(publish-event &events  speechstart (((py-atom datetime.datetime.now)) 2) )
!(publish-event &events speech ((py-dict (("data" "who is John Lennon?"))) 2))
!((py-atom time.sleep) 5)
!(&a1 .say)

!(publish-event &events  speechstart (((py-atom datetime.datetime.now)) 2) )
!(publish-event &events speech ((py-dict (("data" "What is my name?"))) 2))
!((py-atom time.sleep) 5)
!(&a1 .say)

!(&a1 .stop)